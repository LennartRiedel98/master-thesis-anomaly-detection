"""Score-Ablation des LSTM-AE: gleiche Repraesentation, vier Scorings.

Die Rekonstruktions-Diagnostik (lstm_ae_reconstruction.py, methodology
7.12) zeigt: Der LSTM-AE *lernt* Tages-/Wochenmuster (Latent-Raum nach
Tageszeit geordnet, r(Rekonstruktion, Original) ~ 0,97), verliert aber
alle 9 Vergleichsjobs. Dieses Skript testet die daraus abgeleitete
Hypothese, dass die Schwaeche in der Fehler->Score-Abbildung liegt und
nicht in der Architektur: Es bewertet **dasselbe trainierte Modell**
unter vier Score-Varianten (s. models/lstm_ae.py, ``score_mode``):

    window_mse    Fenster-MSE ueber alle Features (Stage-10-Baseline)
    channel_mse   nur Ziel-Kanal (kw_mean/residual/trend)
    last_step_mse nur letzter Zeitschritt
    mahalanobis   EncDec-AD-Scoring (Malhotra u. a. 2016)

Evaluations-Protokoll = exakt Stage 10 (faire Vergleichbarkeit):
Re-Fit mit den besten HPs aus Stage 8, Schwelle = q-Quantil der
sauberen *Validierungs*-Scores je Modus, point-adjusted P/R/F1 +
ROC-/PR-AUC auf den eligiblen Test-Stunden des injizierten Test-Sets.
Die Stage-10-Hauptergebnisse bleiben unangetastet (eigenes Report-CSV).

Interpretation: Steigt F1 unter einem alternativen Scoring deutlich,
ist der Kriteriums-Mismatch kausal belegt ("Score-Reparatur statt
Architekturwechsel"); bleibt F1 niedrig, ist die DL-Schwaeche robuster
belegt. Beides verwertbar in Kap. 4 (Diagnostik) und 5.6.

``--use-saved`` laedt statt des Re-Fits die Stage-7-Checkpoints
(models/<variant>/<wmz>/lstm_ae.pkl) - schneller Smoke-Test, aber
Default-HPs statt Stage-8-Bestwerte -> nur zur Funktionspruefung,
nicht fuer Thesis-Zahlen.

Eingang: outputs/<ds>/parquet/stage6_normalized_<variant>.parquet,
         stage9_injected_<variant>.parquet, stage9_ground_truth.parquet,
         split_assignment.parquet, hpo/best_hparams.json
Ausgabe: outputs/<ds>/reports/lstm_ae_score_ablation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import evaluate, threshold_from_quantile   # noqa: E402
from models.lstm_ae import LSTMAutoencoderDetector         # noqa: E402
from models.registry import WMZ_NAMES, model_features      # noqa: E402
from stage7_train import select_training_rows              # noqa: E402

DEFAULT_DATASET = "gebaeude_a"
VARIANTS = ["raw", "residual", "trend"]
MODES = ["window_mse", "channel_mse", "last_step_mse", "mahalanobis"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--variants", nargs="+", default=None, choices=VARIANTS)
    p.add_argument("--wmz", nargs="+", default=None, choices=WMZ_NAMES)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device fuer Re-Fit/Scoring. Fuer den Re-Fit auf "
                        "dem Mac 'cpu' erzwingen (nn.LSTM-Training auf MPS "
                        "defekt, s. stage10); Inferenz auf MPS ist ok.")
    p.add_argument("--threshold-quantile", type=float, default=0.99,
                   help="Val-Quantil fuer die Schwelle je Modus (wie "
                        "Stage 10). Default: 0.99")
    p.add_argument("--max-train-rows", type=int, default=None, metavar="N")
    p.add_argument("--use-saved", action="store_true",
                   help="Stage-7-Checkpoints laden statt Re-Fit mit "
                        "Stage-8-HPs (nur Smoke-Test, s. Docstring).")
    return p.parse_args()


def gt_kind_col(variant: str, wmz: str) -> str:
    """Name der Ground-Truth-Spalte fuer die Anomalieart einer Schiene.

    Schiene C (trend) zielt auf nicht-stationaere Anomalien, A und B auf
    stationaere.
    """
    return f"gt_{'nonstat' if variant == 'trend' else 'stat'}_{wmz}"


def main() -> None:
    """Vier Score-Varianten am selben trainierten LSTM-AE vergleichen.

    Die Repraesentation bleibt fest, variiert wird nur die Abbildung vom
    Rekonstruktionsfehler auf den Anomalie-Score. Damit laesst sich die aus
    der Rekonstruktions-Diagnostik abgeleitete Vermutung pruefen, dass der
    Bruch im Scoring sitzt und nicht in der Architektur.

    Befund: In einem Teil der Jobs repariert ein anderer Score-Modus das
    Ergebnis deutlich - an die klassischen Detektoren reicht auch die beste
    Variante nicht heran.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_parquet(parquet_dir / "split_assignment.parquet")["split"]
    gt = pd.read_parquet(parquet_dir / "stage9_ground_truth.parquet")
    best_path = out_root / "hpo" / "best_hparams.json"
    best = json.load(open(best_path)) if best_path.is_file() else {}

    val_index = split.index[split == "val"]
    test_index = split.index[split == "test"]

    sel_variants = args.variants or VARIANTS
    sel_wmz = args.wmz or WMZ_NAMES

    rows: list[dict] = []
    print(f"Dataset: {args.dataset}  "
          f"({'Checkpoints (Smoke-Test)' if args.use_saved else 'Re-Fit mit Stage-8-HPs'})")
    print("=" * 78)

    for variant in sel_variants:
        clean = pd.read_parquet(parquet_dir
                                / f"stage6_normalized_{variant}.parquet")
        inj = pd.read_parquet(parquet_dir
                              / f"stage9_injected_{variant}.parquet")
        for wmz in sel_wmz:
            X_clean = model_features(clean, variant, wmz)
            train_idx = select_training_rows(clean, wmz, variant,
                                             args.max_train_rows)

            if args.use_saved:
                ckpt = out_root / "models" / variant / wmz / "lstm_ae.pkl"
                if not ckpt.is_file():
                    print(f"  {variant}/{wmz}: kein Checkpoint - skip")
                    continue
                model = LSTMAutoencoderDetector.load(ckpt)
                if args.device != "auto":
                    model._device = args.device
                # Alt-Checkpoints tragen keine Fehler-Statistik - auf den
                # Trainings-Zeilen nachziehen (kein Leakage).
                if model.err_mu_ is None:
                    model.fit_error_stats(
                        X_clean[model.feature_cols_].loc[train_idx])
            else:
                hp = (best.get(variant, {}).get(wmz, {})
                      .get("lstm_ae", {}).get("hparams", {})) or {}
                if args.device != "auto":
                    hp = {**hp, "device": args.device}
                model = LSTMAutoencoderDetector(**hp).fit(
                    X_clean.loc[train_idx])

            # Ein Pass je Frame liefert alle vier Modi (Schwellen aus der
            # sauberen Validierung, Bewertung auf dem injizierten Test -
            # identisch zu Stage 10).
            comp_val = model.score_components(X_clean.loc[val_index])
            comp_test = (model
                         .score_components(model_features(inj, variant, wmz))
                         .reindex(test_index))

            elig = (~gt.loc[test_index, f"gt_no_data_{wmz}"].astype(bool)
                    & ~gt.loc[test_index,
                              f"gt_known_sensor_issue_{wmz}"].astype(bool))
            elig_idx = test_index[elig.to_numpy()]
            y_true = gt.loc[elig_idx, gt_kind_col(variant, wmz)].notna()

            base_f1 = None
            for mode in MODES:
                thr = threshold_from_quantile(comp_val[mode],
                                              args.threshold_quantile)
                res = evaluate(comp_test.loc[elig_idx, mode], y_true, thr,
                               adjust=True)
                if mode == "window_mse":
                    base_f1 = res["f1"]
                rows.append({"variant": variant, "wmz": wmz,
                             "score_mode": mode,
                             "delta_f1_vs_window": res["f1"] - base_f1,
                             **res})
                print(f"  {variant}/{wmz}  {mode:<13} "
                      f"P={res['precision']:.3f} R={res['recall']:.3f} "
                      f"F1={res['f1']:.3f} "
                      f"(Δ{res['f1'] - base_f1:+.3f})  "
                      f"ROC={res['roc_auc']:.3f} PR={res['pr_auc']:.3f}")
            print()

    out = pd.DataFrame(rows)
    csv_path = reports_dir / "lstm_ae_score_ablation.csv"
    out.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    # Kontext: Klassik-Champions aus Stage 10 zum direkten Vergleich.
    m_path = reports_dir / "stage10_metrics.csv"
    if m_path.is_file():
        m = pd.read_csv(m_path)
        m = m[(m["stratum"] == "overall") & (m["model"] != "lstm_ae")]
        print("\n  Klassik-Champion je Job (Stage 10, overall F1):")
        for (variant, wmz), grp in m.groupby(["variant", "wmz"]):
            b = grp.loc[grp["f1"].idxmax()]
            print(f"    {variant}/{wmz}: {b['model']} F1={b['f1']:.3f}")


if __name__ == "__main__":
    main()
