"""Stage 5 - Train/Val/Test-Split.

Einmalige, zeitlich strikte Aufteilung des stuendlichen Index aus Stage 2.
Gilt identisch fuer alle Detektions-Varianten (A, B, C) und alle Modelle,
damit Vergleichbarkeit gewaehrleistet ist.

Zeitliche Grenzen (gemaess Methodik-Sektion 7.2):

    Train  [2019-11-19 00:00, 2022-11-01 00:00)
    Val    [2022-11-01 00:00, 2023-05-01 00:00)
    Test   [2023-05-01 00:00, 2023-11-19 00:00)   # exklusives Ende aus Stage 1

Der Test-Bereich faellt absichtlich in das Jahr mit der schlechtesten
Datenqualitaet (vgl. Methodik-Sektion 5.3) und wird damit unbeabsichtigt
aber wertvoll "realistischer" als der Trainings-Bereich.

Output:
    outputs/<dataset>/parquet/split_assignment.parquet
        Index:   timestamp (DatetimeIndex, stuendlich)
        Spalte:  split (Categorical: "train" / "val" / "test")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Projekt-Wurzel: src/stage5_split.py -> parents[1] = anomaly-detection-master-thesis-main/
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "gebaeude_a"

# Drei Zeitstempel definieren die Splitgrenzen.
# Konvention: linksgeschlossen, rechtsoffen ([a, b)) - identisch zur
# Stage-1-Clipping-Konvention. Damit gibt es keine Doppelzuordnung an
# den Grenzen, und der erste Zeitstempel des naechsten Splits liegt
# garantiert eine Stunde nach dem letzten Zeitstempel des vorherigen.
TRAIN_START = pd.Timestamp("2019-11-19 00:00:00")
VAL_START = pd.Timestamp("2022-11-01 00:00:00")
TEST_START = pd.Timestamp("2023-05-01 00:00:00")
TEST_END = pd.Timestamp("2023-11-19 00:00:00")   # exklusiv (wie Stage 1)

# Reihenfolge ist relevant: wird als CategoricalDtype-Ordnung
# durchgereicht, sodass spaeter z. B. ``df.sort_values("split")``
# Train vor Val vor Test sortiert.
SPLIT_CATEGORIES = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Name des Unterordners unter outputs/. "
                        f"Default: {DEFAULT_DATASET}")
    return p.parse_args()


def assign_split(index: pd.DatetimeIndex) -> pd.Series:
    """Mappt jeden Zeitstempel auf 'train' / 'val' / 'test'.

    Linksgeschlossene, rechtsoffene Intervalle. Zeitstempel ausserhalb des
    abgedeckten Bereichs (sollte es nach Stage-1-Clipping nicht geben)
    werden als <NA> markiert und am Ende detektiert.
    """
    # object-dtype als Zwischenstation, damit Boolean-Indexing problemlos
    # Strings einsetzen kann. Wird am Ende in Categorical konvertiert.
    labels = pd.Series(pd.NA, index=index, dtype="object")

    # Drei disjunkte Boolean-Masken. Da die Intervalle nicht ueberlappen,
    # darf jedes Element nur in EINER der Masken True sein -> die
    # Reihenfolge der Zuweisungen ist hier eigentlich irrelevant.
    labels[(index >= TRAIN_START) & (index < VAL_START)] = "train"
    labels[(index >= VAL_START) & (index < TEST_START)] = "val"
    labels[(index >= TEST_START) & (index < TEST_END)] = "test"

    # Kategorischer Dtype spart Speicher (kein wiederholter String je Zeile)
    # und ist fuer spaetere Gruppierungen/Plots semantisch klarer.
    return labels.astype(pd.CategoricalDtype(categories=SPLIT_CATEGORIES,
                                             ordered=True))


def main() -> None:
    """Stage 5 ausfuehren: den zeitlichen Train/Val/Test-Schnitt festlegen.

    Der Schnitt erfolgt streng chronologisch und ohne Mischen - bei
    Zeitreihen wuerde eine zufaellige Aufteilung Zukunftswissen ins Training
    tragen. Er wird einmal berechnet und gilt anschliessend fuer alle
    Schienen und alle Modelle, damit die Ergebnisse vergleichbar bleiben.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    stage2_path = out_root / "parquet" / "stage2_hourly.parquet"
    out_path = out_root / "parquet" / "split_assignment.parquet"

    if not stage2_path.is_file():
        raise SystemExit(
            f"Stage-2 nicht gefunden: {stage2_path}\n"
            f"Run: python src/stage2_preprocess.py --dataset {args.dataset}"
        )

    print(f"Dataset: {args.dataset}")
    # columns=[] laedt nur den Index, ohne Wertespalten - Stage 5 braucht
    # ausschliesslich den DatetimeIndex aus Stage 2 als kanonische
    # Stundenraster-Referenz. Spart Memory und I/O.
    stage2 = pd.read_parquet(stage2_path, columns=[])
    print(f"  Stage 2 input: {len(stage2)} Zeitstempel "
          f"({stage2.index.min()} -> {stage2.index.max()})")

    split = assign_split(stage2.index)
    # Sanity: jeder Zeitstempel muss genau einem Split zugeordnet sein.
    # Falls nicht, ist entweder Stage 1 anders geclippt oder der Index
    # enthaelt Zeitstempel ausserhalb der definierten Splitgrenzen.
    n_unassigned = int(split.isna().sum())
    if n_unassigned:
        raise SystemExit(
            f"  {n_unassigned} Zeitstempel ausserhalb der Split-Grenzen "
            f"- Stage-1-Clipping pruefen."
        )

    out_df = split.to_frame(name="split")
    # Konsistenter Index-Name in allen Pipeline-Parquets - vereinfacht
    # spaetere Joins / Reindex-Operationen.
    out_df.index.name = "timestamp"

    # --- Quality-Report ----------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 5 Quality Report")
    print("=" * 70)

    total = len(out_df)
    print(f"\n  Gesamt-Stunden: {total}")
    print(f"\n  {'Split':<6} {'Start':<20} {'Ende (inkl.)':<20} "
          f"{'Stunden':>9} {'Anteil':>8}")
    print(f"  {'-' * 6} {'-' * 19} {'-' * 19} {'-' * 9} {'-' * 8}")
    for name in SPLIT_CATEGORIES:
        mask = out_df["split"] == name
        sub = out_df.index[mask]
        n = len(sub)
        pct = 100 * n / total
        print(f"  {name:<6} {str(sub.min()):<20} {str(sub.max()):<20} "
              f"{n:>9} {pct:>7.2f}%")

    # Sanity-Assert: die drei Splits muessen den Gesamtindex partitionieren
    # (disjunkt + vollstaendig). Da assign_split disjunkte Intervalle setzt,
    # reicht es, die Summe der Mengenmaechtigkeiten gegen die Gesamtzahl
    # zu checken.
    summed = sum((out_df["split"] == n).sum() for n in SPLIT_CATEGORIES)
    assert summed == total, f"Split-Summe {summed} != Gesamt {total}"
    print("\n  Partition vollstaendig (Summe == Gesamt-Stunden).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
