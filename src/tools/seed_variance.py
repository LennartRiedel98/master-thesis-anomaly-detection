"""Seed-Varianz-Analyse fuer den LSTM-AE-HPO-Lauf mit ``--hpo-seeds N``.

Bewertet, ob die in Stage 8 getroffenen HP-Entscheidungen statistisch
belastbar sind oder im Rauschen der Initialisierungs-/Sampling-Varianz
untergehen (Reimers und Gurevych 2017; Bouthillier u. a. 2021).

**Was es tut.** Liest ``hpo_log.csv`` aus dem aktuellen Output-
Verzeichnis, filtert auf die LSTM-AE-Zeilen und gruppiert nach
``(variant, wmz, hp_window_size, hp_hidden_size, hp_n_layers,
hp_learning_rate, hp_epochs)``. Mit ``--hpo-seeds 3`` in Stage 8
produziert jede HP-Konfiguration **drei** Zeilen (eine pro Seed) statt
einer. Wir aggregieren ueber die Seeds zu Mean/Std/Min/Max der Scores
und werten dann fuer jeden Job aus, **wie sicher die Best-HP-Wahl
wirklich war**.

**Was die Ausgabe sagt.** Pro Job:

* ``best_hp`` — die HP-Kombination mit hoechstem Seed-Mean.
* ``best_mean`` +/- ``best_std`` — gemittelter Score der Best-HP samt
  Inter-Seed-Streuung.
* ``runner_up_mean`` — Score der zweitbesten HP-Kombination.
* ``gap = best_mean - runner_up_mean`` — Abstand der Best- von der
  Runner-up-HP.
* ``robust`` = ``gap > 2 * best_std`` (~2-Sigma-Regel) — grobe
  Heuristik, ob der Best-vs-Runner-up-Abstand groesser als die
  Seed-Unsicherheit ist.

Wenn ``robust = False`` in vielen Jobs, ist die HP-Wahl ueber den
Seed-Mittelwert zwar deterministisch, aber **inhaltlich beliebig** —
ein Run mit einem anderen Seed-Tripel haette eine andere HP gewaehlt.
Das ist genau der Punkt, der in Ergebnisbericht §15.7 als „Kriteriums-
Mismatch" angerissen wurde und mit dem konkreten ``robust``-Vektor
empirisch belegbar wird.

Erwartet wird der Lauf ``python src/stage8_hpo.py --models lstm_ae
--lstm-strategy gridwh --device cuda --hpo-metric pr_auc --hpo-seeds 3``
Ohne Seed-Averaging
(``--hpo-seeds 1``, Default) gibt es **nur eine** Zeile pro Config —
dann sind Std/Gap-Spalten alle 0/NaN und die ``robust``-Spalte verliert
ihre Aussage. Das Skript faengt das ab und meldet es.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

# Spalten, die eine HP-Konfiguration *eindeutig* bestimmen. Stage 8
# schreibt die HP-Namen ohne Praefix in ``hpo_log.csv`` (siehe Header
# der Datei). Spalten, die unter ``--hpo-seeds N`` *innerhalb* einer
# Config variieren (z. B. ein Init-Seed, wenn er protokolliert waere),
# gehoeren bewusst NICHT hier rein - sonst gruppiert man jede
# Seed-Wiederholung in einen eigenen Bucket und der Seed-Mittelwert
# verschwindet.
LSTM_HP_COLS = [
    "window_size", "hidden_size", "n_layers",
    "learning_rate", "epochs",
]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    p.add_argument("--sigma-factor", type=float, default=2.0,
                   help="Faktor k fuer die robust-Regel gap > k*std. "
                        "Default 2.0 (grob 95%% bei normalverteilter "
                        "Seed-Streuung).")
    return p.parse_args()


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Gruppiert nach (variant, wmz, HP-Kombi) und aggregiert ``score`` ueber die Seeds.

    Spalten der Hp-Kombi, die im Log gar nicht vorkommen (z. B. ein
    Klassik-Job wurde versehentlich miteingelesen), werden vorher
    weggefiltert. Eine Zeile mit allen-NaN-HPs wuerde sonst zu einem
    Pseudo-Bucket fuehren.
    """
    # NaN-Filterung: nur Zeilen mit gesetzten LSTM-HPs (= LSTM-AE-Zeilen).
    sub = df.dropna(subset=LSTM_HP_COLS, how="all")
    if sub.empty:
        return pd.DataFrame()

    grp_cols = ["variant", "wmz", *LSTM_HP_COLS]
    agg = (sub.groupby(grp_cols)["score"]
              .agg(["mean", "std", "min", "max", "count"])
              .reset_index())
    # Std=NaN tritt bei count=1 auf (single seed). Wir setzen das auf 0,
    # damit die spaetere robust-Logik einfacher faellt - dokumentieren das
    # aber in der CSV ueber die separate ``count``-Spalte.
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def robustness_per_job(agg: pd.DataFrame, sigma_factor: float) -> pd.DataFrame:
    """Pro (variant, wmz) Best/Runner-up + Robustheits-Heuristik berechnen."""
    rows = []
    for (variant, wmz), grp in agg.groupby(["variant", "wmz"]):
        # Nach Seed-Mittel absteigend sortieren - dadurch ist Index 0 die
        # Best-HP, Index 1 der unmittelbar dahinterliegende Runner-up.
        ranked = grp.sort_values("mean", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        runner = ranked.iloc[1] if len(ranked) > 1 else None
        gap = float(best["mean"] - runner["mean"]) if runner is not None else float("nan")
        # 2-Sigma-Regel: liegt der Abstand zum Zweitbesten oberhalb der
        # Seed-Unsicherheit der Besten? Falls ja, ist die Wahl
        # statistisch klar. Falls nein, lebt sie im Rauschen.
        threshold = sigma_factor * float(best["std"])
        robust = (gap > threshold) if not np.isnan(gap) else False

        # Best-HP als String, damit die CSV in *einer* Zeile lesbar ist.
        best_hp = " ".join(f"{c}={best[c]:g}" for c in LSTM_HP_COLS)
        rows.append({
            "variant": variant, "wmz": wmz,
            "best_hp": best_hp,
            "best_mean": round(float(best["mean"]), 4),
            "best_std": round(float(best["std"]), 4),
            "best_min": round(float(best["min"]), 4),
            "best_max": round(float(best["max"]), 4),
            "n_seeds": int(best["count"]),
            "runner_up_mean": round(float(runner["mean"]), 4)
                              if runner is not None else float("nan"),
            "gap": round(gap, 4) if not np.isnan(gap) else float("nan"),
            "robust_2sigma": bool(robust),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Streuung der LSTM-AE-Ergebnisse ueber verschiedene Startwerte auswerten.

    Liest das HPO-Protokoll, filtert die LSTM-AE-Zeilen und gruppiert nach
    Hyperparameter-Kombination. Die Frage dahinter: Sind die in Stage 8
    getroffenen Entscheidungen belastbar, oder liegt der Abstand zwischen
    zwei Kombinationen unter dem Rauschen der Initialisierung? Das Ergebnis
    ist als Limitation in der Arbeit gefuehrt.
    """
    args = parse_args()
    log_path = (ROOT / "outputs" / args.dataset / "hpo" / "hpo_log.csv")
    if not log_path.is_file():
        raise SystemExit(f"hpo_log.csv nicht gefunden: {log_path}")
    df = pd.read_csv(log_path)
    df = df[df["model"] == "lstm_ae"].copy()
    if df.empty:
        raise SystemExit("Keine LSTM-AE-Zeilen in hpo_log.csv "
                         "(lief Stage 8 ueberhaupt fuer lstm_ae?).")

    agg = aggregate_seeds(df)
    if agg.empty:
        raise SystemExit("Keine LSTM-HP-Spalten gefunden - hpo_log hat "
                         "ein unerwartetes Schema.")

    out = robustness_per_job(agg, args.sigma_factor)

    # Seed-Coverage pro Best-HP berechnen: wenn die Best-HP der Mehrheit
    # der Jobs nur mit *einem* Seed gemessen wurde, sind die std/gap/
    # robust-Aussagen trivial (std=0 -> robust immer True). Wir warnen
    # ueber die Best-HP, nicht ueber den globalen max-count, weil der
    # globale max auch >1 sein kann, ohne dass die Best-HP davon
    # profitiert (z. B. wenn nur die zweitbeste Config zufaellig
    # mehrfach vorkommt).
    median_n_seeds_best = int(out["n_seeds"].median())
    if median_n_seeds_best < 2:
        print("\nWARNUNG: die Best-HP der meisten Jobs wurde nur mit "
              f"einem Seed gemessen (Median n_seeds = {median_n_seeds_best}). "
              "Die Std-/Gap-/Robustheits-Spalten sind dann trivial "
              "(std=0 -> robust immer True). Setze --hpo-seeds 2 oder "
              "mehr beim naechsten Stage-8-Lauf, damit diese Diagnose "
              "Substanz bekommt.")
    out_path = (ROOT / "outputs" / args.dataset / "reports"
                / "lstm_seed_variance.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Konsolen-Zusammenfassung: die Headline-Aussage fuer den Bericht.
    print(f"\nSeed-Varianz pro Job (Median n_seeds = {median_n_seeds_best}):")
    print(out.to_string(index=False))
    n_total = len(out)
    n_robust = int(out["robust_2sigma"].sum())
    print(f"\n  Robust nach 2-Sigma-Regel: {n_robust}/{n_total} Jobs.")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
