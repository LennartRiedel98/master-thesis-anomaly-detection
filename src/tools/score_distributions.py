"""Score-Verteilungen LSTM-AE vs. Klassik-Champion auf dem Test-Set.

Visualisiert pro (Variante x WMZ), wie die Anomalie-Scores des LSTM-AE
und des jeweiligen Klassik-Champions (IForest auf raw, LOF auf
residual, PELT auf trend) zwischen den Ground-Truth-Klassen
``normal`` und ``anomal`` verteilt sind. Damit wird sichtbar, **warum**
ein Modell die F1-Werte aus Stage 10 liefert, die es liefert.

Drei diagnostisch relevante Muster, die der Plot direkt sichtbar
macht (siehe Ergebnisbericht §15.7 zur Motivation):

* **Score-Inversion** — die Anomalie-Verteilung liegt *links* von der
  Normal-Verteilung (= Anomalien rekonstruieren *besser*, niedrigerer
  MSE). Das ist genau das Muster, das den AUC unter 0,5 erzeugt, das
  bisher nur indirekt in der ROC-AUC-Spalte erkennbar ist.
* **Schwacher Overlap** — die Verteilungen ueberlappen stark; das
  Modell trennt schlecht und die F1 bleibt unabhaengig vom Schwellwert
  niedrig.
* **Falsche Schwelle** — die Verteilungen sind getrennt, aber das
  in Stage 10 verwendete Validierungs-Quantil (Default 0,99) liegt im
  falschen Bereich. Das waere ein **fixbarer** Defekt, kein
  Modellversagen.

Pro (Variante x WMZ) zeichnen wir zwei normalisierte Histogramme
(KDE-aehnliche Glaettung waere irrefuehrend bei <100 anomalen Stunden
in manchen Jobs), uebereinander gelegt: einmal LSTM-AE, einmal der
Klassik-Champion. Beide Score-Reihen werden zur Vergleichbarkeit per
WMZ auf das 1.- bis 99.-Perzentil geclippt und auf [0, 1] skaliert
(Min-Max), damit die Form, nicht die absolute Skala, im Vordergrund
steht.

Inputs:
* ``outputs/<ds>/models/<variant>/<wmz>/{lstm_ae, iforest, lof, pelt}.pkl``
* ``outputs/<ds>/parquet/stage9_injected_<variant>.parquet``
* ``outputs/<ds>/parquet/stage9_ground_truth.parquet``
* ``outputs/<ds>/parquet/split_assignment.parquet`` (Test-Index)

Output:
* ``outputs/<ds>/figures/score_verteilungen.png`` (3x3 Grid, eine
  Subplot pro Variante x WMZ; rote vs. blaue Verteilung = anomal vs.
  normal; gestrichelte Linie = Stage-10-Schwellwert).
* Konsolen-Logging der drei Diagnose-Befunde pro Job
  (``inversion / overlap / threshold-shift``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

# src/ in den PYTHONPATH legen, damit ``from models...`` aus diesem
# tools/-Skript funktioniert wie in den Stages selber.
sys.path.insert(0, str(ROOT / "src"))

from evaluation import threshold_from_quantile   # noqa: E402
from models.iforest import IForestDetector       # noqa: E402
from models.lof import LOFDetector               # noqa: E402
from models.lstm_ae import LSTMAutoencoderDetector  # noqa: E402
from models.pelt import PELTDetector             # noqa: E402
from models.registry import model_features       # noqa: E402

# Pro Variante: welches Klassik-Modell ist der Champion (siehe
# Ergebnisbericht §11.2)? Diese Wahl ist hier hart kodiert, weil die
# Diagnose explizit die *besten* klassischen Detektoren mit dem
# LSTM-AE konfrontiert - nicht alle Klassiker.
CHAMPION_CLASS = {
    "raw":      ("iforest",  IForestDetector),
    "residual": ("lof",      LOFDetector),
    "trend":    ("pelt",     PELTDetector),
}
WMZ_NAMES = ["wmz_1", "wmz_2", "wmz_3"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    p.add_argument("--threshold-quantile", type=float, default=0.99,
                   help="Quantil-Schwelle wie in Stage 10 (Default 0,99).")
    p.add_argument("--bins", type=int, default=40,
                   help="Anzahl Histogramm-Bins pro Subplot.")
    return p.parse_args()


def gt_kind_col(variant: str, wmz: str) -> str:
    """Schiene wie in Stage 10: raw/residual -> stationaer, trend -> nicht-stat."""
    return f"gt_{'nonstat' if variant == 'trend' else 'stat'}_{wmz}"


def load_model(variant: str, wmz: str, model_name: str, out_root: Path):
    """Laedt das passende Modell-.pkl. Nutzt direkt die Klassen-load()."""
    path = out_root / "models" / variant / wmz / f"{model_name}.pkl"
    if not path.is_file():
        raise SystemExit(f"Modell fehlt: {path}")
    if model_name == "lstm_ae":
        return LSTMAutoencoderDetector.load(path)
    if model_name == "iforest":
        return IForestDetector.load(path)
    if model_name == "lof":
        return LOFDetector.load(path)
    if model_name == "pelt":
        return PELTDetector.load(path)
    raise ValueError(f"Unbekanntes Modell: {model_name}")


def normalize_clip(scores: np.ndarray) -> np.ndarray:
    """Per-WMZ Min-Max-Skalierung nach Clipping auf [P1, P99].

    Macht die Verteilungen ueber Modelle vergleichbar (LSTM-MSE und
    IForest-Score liegen sonst in komplett verschiedenen Bereichen),
    ohne extreme Ausreisser die Plot-Achse dehnen zu lassen.
    """
    s = scores[~np.isnan(scores)]
    if s.size == 0:
        return scores
    lo, hi = np.percentile(s, [1, 99])
    clipped = np.clip(scores, lo, hi)
    rng = hi - lo
    if rng <= 0:
        return np.zeros_like(scores)
    return (clipped - lo) / rng


def diagnose(scores: np.ndarray, y_true: np.ndarray,
             threshold_norm: float) -> str:
    """Klassifiziert den dominanten Befund: inversion / overlap / threshold-shift / ok.

    Mit den normalisierten Scores in [0, 1]:
    - inversion: Mean(anomal) < Mean(normal) - 0,05 (klar invertiert)
    - threshold-shift: getrennte Means (>0,1), aber Schwelle liegt nicht
      zwischen ihnen
    - overlap: Mean-Abstand <0,05
    - ok: getrennt + Schwelle dazwischen
    """
    if scores.size == 0 or y_true.size == 0:
        return "no-data"
    a = scores[y_true.astype(bool)]
    n = scores[~y_true.astype(bool)]
    if a.size == 0 or n.size == 0:
        return "single-class"
    ma, mn = float(np.nanmean(a)), float(np.nanmean(n))
    gap = ma - mn
    if gap < -0.05:
        return "inversion"
    if abs(gap) < 0.05:
        return "overlap"
    # gap >= 0.05: Anomalien rangieren ueber Normal. Schwelle muss
    # dazwischen liegen, sonst threshold-shift.
    if mn < threshold_norm < ma:
        return "ok"
    return "threshold-shift"


def main() -> None:
    """Score-Verteilungen von LSTM-AE und klassischem Champion gegenueberstellen.

    Zeigt je Schiene und Zaehler, wie sich die Scores auf die
    Ground-Truth-Klassen normal und anomal verteilen. Damit wird sichtbar,
    *warum* ein Modell die Kennzahlen liefert, die es liefert: ob die Klassen
    ueberlappen, ob die Rangfolge invertiert ist oder ob nur die Schwelle an
    der falschen Stelle liegt.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    fig_dir = out_root / "figures"

    split = pd.read_parquet(parquet_dir / "split_assignment.parquet")["split"]
    gt = pd.read_parquet(parquet_dir / "stage9_ground_truth.parquet")
    test_idx = split.index[split == "test"]
    val_idx = split.index[split == "val"]

    # 3 Varianten x 3 WMZ = 9 Subplots, 2 Spalten je Subplot (LSTM + Champ).
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    findings: list[dict] = []
    variants = ["raw", "residual", "trend"]

    for row, variant in enumerate(variants):
        clean = pd.read_parquet(parquet_dir / f"stage6_normalized_{variant}.parquet")
        inj = pd.read_parquet(parquet_dir / f"stage9_injected_{variant}.parquet")
        champ_name, _ = CHAMPION_CLASS[variant]

        for col, wmz in enumerate(WMZ_NAMES):
            ax = axes[row, col]
            # Eligibilitaet wie in Stage 10: ohne no_data, ohne known
            # sensor issue. So bleibt der Plot mit den Tabellen aus
            # §11.2 konsistent.
            elig = (~gt.loc[test_idx, f"gt_no_data_{wmz}"].astype(bool)
                    & ~gt.loc[test_idx, f"gt_known_sensor_issue_{wmz}"]
                       .astype(bool))
            elig_idx = test_idx[elig.to_numpy()]
            y_true = gt.loc[elig_idx, gt_kind_col(variant, wmz)].notna().to_numpy()

            # LSTM-AE: scort auf der vollen injizierten Reihe (wegen
            # Fenster-Vorlauf), dann auf eligible Test-Stunden slicen.
            lstm = load_model(variant, wmz, "lstm_ae", out_root)
            lstm_scores_full = lstm.score(inj).reindex(elig_idx).to_numpy(dtype=float)

            # Klassik-Champion: scort auf der schon normalisierten
            # Feature-Tabelle (Eingabe identisch zu Stage 10).
            X_champ = model_features(clean, variant, wmz)
            champ = load_model(variant, wmz, champ_name, out_root)
            champ_scores_full = champ.score(X_champ.loc[elig_idx]).to_numpy(dtype=float)

            # Schwellwert wie in Stage 10: 0.5 fuer PELT (binaer),
            # sonst q-Quantil der sauberen Validation-Scores.
            if champ_name == "pelt":
                champ_thresh = 0.5
            else:
                X_clean_full = model_features(clean, variant, wmz)
                champ_thresh = threshold_from_quantile(
                    champ.score(X_clean_full.loc[val_idx]),
                    args.threshold_quantile,
                )
            X_clean_full = model_features(clean, variant, wmz)
            lstm_thresh = threshold_from_quantile(
                lstm.score(X_clean_full.loc[val_idx]),
                args.threshold_quantile,
            )

            # Per-Modell normalisieren (siehe Docstring). Wir wenden
            # dieselbe Min-Max-Transformation auf die Schwelle an, damit
            # die gestrichelte Linie im Plot sinnvoll bleibt.
            def _norm(arr: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
                """Scores auf [0, 1] bringen, damit zwei Detektoren vergleichbar werden.

                Skaliert wird am 1- und 99-Prozent-Quantil der Referenzverteilung statt
                am Minimum und Maximum: Ein einzelner Extremwert wuerde die Skala sonst
                so stauchen, dass von der eigentlichen Verteilung nichts mehr zu sehen
                ist. Der zweite Rueckgabewert ist die mitskalierte Schwelle.
                """
                ref_valid = ref[~np.isnan(ref)]
                if ref_valid.size == 0:
                    return arr, 0.5
                lo, hi = np.percentile(ref_valid, [1, 99])
                rng = hi - lo if hi > lo else 1.0
                return np.clip((arr - lo) / rng, 0, 1), float(np.clip((champ_thresh - lo) / rng, 0, 1))

            lstm_n = normalize_clip(lstm_scores_full)
            champ_n = normalize_clip(champ_scores_full)
            lstm_t_n = normalize_clip(np.array([lstm_thresh,
                                                *lstm_scores_full[~np.isnan(lstm_scores_full)]]))[0]
            champ_t_n = normalize_clip(np.array([champ_thresh,
                                                 *champ_scores_full[~np.isnan(champ_scores_full)]]))[0]

            # Histogramme zeichnen. ``density=True`` macht beide
            # Modelle vergleichbar trotz ungleich vieler Punkte.
            common = dict(bins=args.bins, density=True, alpha=0.55, range=(0, 1))
            ax.hist(lstm_n[y_true], color="tab:red", label="LSTM-AE anomal",
                    histtype="stepfilled", **common)
            ax.hist(lstm_n[~y_true], color="tab:blue", label="LSTM-AE normal",
                    histtype="stepfilled", **common)
            ax.hist(champ_n[y_true], color="darkred", label=f"{champ_name} anomal",
                    histtype="step", linewidth=1.5, **common)
            ax.hist(champ_n[~y_true], color="darkblue", label=f"{champ_name} normal",
                    histtype="step", linewidth=1.5, **common)
            ax.axvline(lstm_t_n, color="tab:red", linestyle=":", linewidth=1,
                       label="LSTM-Schwelle")
            ax.axvline(champ_t_n, color="darkred", linestyle="--", linewidth=1,
                       label=f"{champ_name}-Schwelle")
            ax.set_title(f"{variant}/{wmz} (vs. {champ_name})", fontsize=10)
            ax.set_xlabel("normalisierter Score [0, 1]")
            ax.set_ylabel("Dichte")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

            # Diagnose-Befunde fuer beide Modelle.
            lstm_diag = diagnose(lstm_n, y_true, lstm_t_n)
            champ_diag = diagnose(champ_n, y_true, champ_t_n)
            findings.append({"variant": variant, "wmz": wmz,
                             "lstm_diag": lstm_diag,
                             "champion_diag": f"{champ_name}:{champ_diag}"})

    fig.suptitle("Score-Verteilungen Test-Set: LSTM-AE vs. Klassik-Champion "
                 "(Davis und Goadrich 2006 zur Imbalance-Sensitivitaet)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "score_verteilungen.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    diag_df = pd.DataFrame(findings)
    diag_path = (out_root / "reports" / "score_distribution_diagnose.csv")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(diag_path, index=False)

    print(f"\nDiagnose-Befunde pro Job:")
    print(diag_df.to_string(index=False))
    print(f"\n  Wrote {out_path}")
    print(f"  Wrote {diag_path}")


if __name__ == "__main__":
    main()
