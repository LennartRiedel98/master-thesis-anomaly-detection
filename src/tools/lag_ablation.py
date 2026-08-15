"""Seiten-Ablation: bringen Lag-Features dem IsolationForest einen Mehrwert?

**Ergebnisneutral** — schreibt NICHTS nach outputs/ (nur Konsole), tastet also
den eingefrorenen Stage-10-Massstab nicht an. Vergleicht IsolationForest auf
raw/wmz_1 OHNE vs. MIT Lag-Features (x_{t-1}, x_{t-24}, x_{t-168}) bei
*identischen* besten HPs aus Stage 8, am selben Lineal (Schwelle = Val-q99,
point-adjusted F1 + Recall je Anomalietyp).

Zweck: empirisch pruefen, ob explizite Lags den Punktdetektoren etwas bringen,
das die vorhandenen Rolling-Features (mean/std/deviation) nicht schon liefern -
bevor man den teuren vollen Pipeline-Re-Run (inkl. GPU-LSTM-AE) startet.

Aufruf:  python src/tools/lag_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import evaluate, threshold_from_quantile, segments   # noqa: E402
from models.registry import model_features                          # noqa: E402
from models.iforest import IForestDetector                          # noqa: E402
from stage7_train import select_training_rows                       # noqa: E402

DATASET = "demo_synthetic"
VARIANT = "raw"
WMZ = "wmz_1"
LAGS = (1, 24, 168)        # 1 h, 1 Tag, 1 Woche
Q = 0.99


def add_lags(X: pd.DataFrame, frame: pd.DataFrame, with_lags: bool) -> pd.DataFrame:
    """Haengt Lag-Spalten des (normalisierten) Signals an die Feature-Matrix.

    Lags werden auf demselben Frame berechnet, aus dem die Features stammen
    (clean bzw. injiziert), damit der Test die gestoerte Trajektorie sieht -
    konsistent zur Stage-9-Logik, die abgeleitete Features auf dem gestoerten
    Signal neu berechnet.
    """
    X = X.copy()
    if with_lags:
        sig = frame[f"{WMZ}_kw_mean"]
        for k in LAGS:
            X[f"{WMZ}_lag_{k}h"] = sig.shift(k)
    return X


def per_type_recall(scores: pd.Series, thr: float, kind: pd.Series) -> dict:
    """Point-adjusted Recall je Anomalietyp (Segment erkannt = irgendein Punkt > thr)."""
    s = scores.reindex(kind.index).to_numpy(dtype=float)
    hit = np.where(np.isnan(s), False, s > thr)
    out = {}
    for v in sorted(kind.dropna().unique()):
        segs = segments((kind == v).to_numpy())
        det = sum(1 for a, b in segs if hit[a:b].any())
        out[v] = (det, len(segs))
    return out


def main() -> None:
    """Pruefen, ob Lag-Merkmale dem Isolation Forest etwas bringen.

    Vergleicht denselben Detektor mit und ohne die Merkmale x(t-1), x(t-24)
    und x(t-168), bei identischen Hyperparametern und am selben Massstab.
    Das Ergebnis ist die Begruendung dafuer, auf Lag-Merkmale zu verzichten:
    Die rollierenden Statistiken decken den zeitlichen Kontext bereits ab.

    Schreibt bewusst nichts nach ``outputs/`` - der eingefrorene
    Stage-10-Massstab bleibt unberuehrt, die Ausgabe geht nur auf die Konsole.
    """
    out = ROOT / "outputs" / DATASET
    pq = out / "parquet"
    clean = pd.read_parquet(pq / f"stage6_normalized_{VARIANT}.parquet")
    inj = pd.read_parquet(pq / f"stage9_injected_{VARIANT}.parquet")
    gt = pd.read_parquet(pq / "stage9_ground_truth.parquet")
    split = pd.read_parquet(pq / "split_assignment.parquet")["split"]
    val_idx = split.index[split == "val"]
    test_idx = split.index[split == "test"]

    best = json.load(open(out / "hpo" / "best_hparams.json"))
    hp = best[VARIANT][WMZ]["iforest"]["hparams"]

    kind = gt.loc[test_idx, f"gt_stat_{WMZ}"]
    elig = (~gt.loc[test_idx, f"gt_no_data_{WMZ}"].astype(bool)
            & ~gt.loc[test_idx, f"gt_known_sensor_issue_{WMZ}"].astype(bool))
    ei = test_idx[elig.to_numpy()]

    print("=" * 68)
    print(f"Lag-Ablation  IForest  {VARIANT}/{WMZ}")
    print(f"  HP={hp}  | Lags={LAGS}  | Schwelle=Val-q{Q}  | ergebnisneutral")
    print("=" * 68)

    rows = []
    for with_lags in (False, True):
        tag = "MIT Lags" if with_lags else "OHNE Lags (Baseline)"
        Xc = add_lags(model_features(clean, VARIANT, WMZ), clean, with_lags)
        tr = select_training_rows(clean, WMZ, VARIANT)
        model = IForestDetector(**hp).fit(Xc.loc[tr])
        thr = threshold_from_quantile(model.score(Xc.loc[val_idx]), Q)
        Xi = add_lags(model_features(inj, VARIANT, WMZ), inj, with_lags)
        sc = model.score(Xi).reindex(test_idx)
        ov = evaluate(sc.loc[ei], kind.loc[ei].notna(), thr, adjust=True)
        pt = per_type_recall(sc, thr, kind)
        rows.append((tag, len(Xc.columns), ov, pt))

    for tag, ncols, ov, pt in rows:
        print(f"\n--- {tag}  ({ncols} Features) ---")
        print(f"  overall  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
              f"F1={ov['f1']:.3f}  ROC-AUC={ov['roc_auc']:.3f}  PR-AUC={ov['pr_auc']:.3f}")
        print("  Recall/Typ: " + "  ·  ".join(f"{k} {d}/{n}" for k, (d, n) in pt.items()))

    # Delta-Zusammenfassung
    base, lag = rows[0][2], rows[1][2]
    print("\n" + "-" * 68)
    print(f"  Delta (MIT - OHNE):  F1 {lag['f1'] - base['f1']:+.3f}   "
          f"ROC-AUC {lag['roc_auc'] - base['roc_auc']:+.3f}   "
          f"PR-AUC {lag['pr_auc'] - base['pr_auc']:+.3f}")
    print("-" * 68)


if __name__ == "__main__":
    main()
