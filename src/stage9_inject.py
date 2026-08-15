"""Stage 9 - Synthetische Anomalie-Injektion in das Test-Set.

Erzeugt das finale, waehrend HPO nie gesehene Evaluations-Signal. Injiziert
(mit eigenem Seed, != Validierungs-Seed aus Stage 8) Anomalien in die
Test-Stunden und persistiert sowohl die injizierten, normalisierten
Feature-Tabellen je Variante als auch eine vollstaendige Ground-Truth.

Es werden die *vollen* Reihen gespeichert (nur Test-Zeilen sind gestoert):
So hat der LSTM-Autoencoder beim Scoren der ersten Test-Stunden noch
sauberen Vor-Kontext fuer seine gleitenden Fenster.

Ground-Truth-Spalten (Methodik 7.6), je WMZ:
    gt_stat_<wmz> / gt_stat_label_<wmz>        - stationäre Anomalien (B-Kat.)
    gt_nonstat_<wmz> / gt_nonstat_label_<wmz>  - nicht-stationäre (B-Kat.)
    gt_known_sensor_issue_<wmz>                - Kategorie A (Stage-2-Flags)
    gt_no_data_<wmz>                           - kein valides Signal (ausschliessen)
und global:
    gt_known_regulatory                        - Kategorie C (+/-N Tage um EnSikuMaV)

Output:
    outputs/<ds>/parquet/stage9_injected_<variant>.parquet
    outputs/<ds>/parquet/stage9_ground_truth.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from injection_apply import build_injected
from models.registry import WMZ_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "demo_synthetic"

# Test-Seed - bewusst verschieden vom Validierungs-Seed in Stage 8.
TEST_SEED = 99

# EnSikuMaV (Energiesicherungsmassnahmenverordnung) trat am 2022-09-01 in
# Kraft. Liegt im Trainings-Zeitraum; fuer das Test-Set (Mai-Nov 2023)
# daher leer - die Spalte existiert fuer die Variante-C-Auswertung ueber
# die volle Reihe und fuer Vollstaendigkeit der Kategorie-C-Ground-Truth.
REGULATORY_EVENTS = {"EnSikuMaV": pd.Timestamp("2022-09-01")}
REGULATORY_WINDOW_DAYS = 14


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    return p.parse_args()


def main() -> None:
    """Stage 9 ausfuehren: Anomalien ins Test-Set injizieren, Ground-Truth schreiben.

    Der Startwert unterscheidet sich bewusst von dem in Stage 8 - sonst
    waeren die Hyperparameter auf genau die Anomalien optimiert, an denen
    spaeter gemessen wird. Gespeichert werden die vollstaendigen Reihen (nur
    die Test-Zeilen sind gestoert) samt punktgenauer Ground-Truth.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    scaler_dir = out_root / "scalers"

    split = pd.read_parquet(parquet_dir / "split_assignment.parquet")["split"]

    print(f"Dataset: {args.dataset}")
    print(f"  Injiziere Anomalien in das Test-Set (Seed {TEST_SEED}) ...")
    frames, gt = build_injected(parquet_dir, scaler_dir, split, "test", TEST_SEED)

    # --- Kategorie-A- und no-data-Spalten aus den Roh-Features ergaenzen ----
    raw = pd.read_parquet(parquet_dir / "stage4_features_raw.parquet")
    for wmz in WMZ_NAMES:
        flag = f"{wmz}_was_flagged"
        gt[f"gt_known_sensor_issue_{wmz}"] = (
            raw[flag].fillna(False).astype(bool) if flag in raw else False)
        gt[f"gt_no_data_{wmz}"] = raw[f"{wmz}_kw_mean"].isna()

    # --- Kategorie C: regulatorische Fenster --------------------------------
    reg = pd.Series(False, index=gt.index)
    for _name, ts in REGULATORY_EVENTS.items():
        lo = ts - pd.Timedelta(days=REGULATORY_WINDOW_DAYS)
        hi = ts + pd.Timedelta(days=REGULATORY_WINDOW_DAYS)
        reg |= (gt.index >= lo) & (gt.index <= hi)
    gt["gt_known_regulatory"] = reg

    # --- Report -------------------------------------------------------------
    test_mask = (split.reindex(gt.index) == "test").to_numpy()
    print("\n" + "=" * 70)
    print("Stage 9 - Injektions-Report (nur Test-Stunden)")
    print("=" * 70)
    for wmz in WMZ_NAMES:
        stat = gt.loc[test_mask, f"gt_stat_{wmz}"].dropna()
        nonstat = gt.loc[test_mask, f"gt_nonstat_{wmz}"].dropna()
        print(f"\n  {wmz}:")
        print(f"    stationär     : {len(stat):>5} Stunden, "
              f"Typen {sorted(stat.unique().tolist())}")
        print(f"    nicht-stationär: {len(nonstat):>5} Stunden, "
              f"Typen {sorted(nonstat.unique().tolist())}")

    # --- Persistenz ---------------------------------------------------------
    for variant, frame in frames.items():
        frame.index.name = "timestamp"
        path = parquet_dir / f"stage9_injected_{variant}.parquet"
        frame.to_parquet(path)
        print(f"\n  Wrote {path}  {frame.shape}")
    gt.index.name = "timestamp"
    gt_path = parquet_dir / "stage9_ground_truth.parquet"
    gt.to_parquet(gt_path)
    print(f"  Wrote {gt_path}  {gt.shape}")


if __name__ == "__main__":
    main()
