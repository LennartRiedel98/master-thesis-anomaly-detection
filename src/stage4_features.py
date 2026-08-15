"""Stage 4 - Feature Engineering pro Detektions-Variante.

Erzeugt zwei Feature-Tabellen mit unterschiedlichem Umfang, jeweils auf
dem stuendlichen Index aus Stage 2:

  Variante A (Rohdaten + volle Features):
    Eingang: wmz_N_kw_mean (Stunden-Mittelwert von kW)
    Pro Zaehler: Signal + 5 rollierende Statistik-Features
    Geteilt:     6 zyklische Zeit-Encodings + 4 binaere Flags
                 + 2 Wetter-Features
    Gedacht fuer Modelle, die Saisonalitaet aus Features lernen sollen.

  Variante B (MSTL-Residuum + minimaler Kontext):
    Eingang: wmz_N_residual aus Stage 3
    Pro Zaehler: Residuum + Trend (Skalen-Kontext) + rolling_std_24h
                 (fuer Plateau-Detektion via Variance-Verlust)
    Geteilt:     is_holiday (irregulaer, nicht von MSTL erfasst)
                 + temperature (kurzfristige Wetter-Variation)
    Cyclic-Encodings und Wochentag-Flags sind redundant zu den MSTL-
    Saisonalitaeten und werden bewusst NICHT mitgegeben.

Begruendung der unterschiedlichen Feature-Saetze:
  - Variante B testet, ob STL-Vorverarbeitung den Modellen Arbeit
    abnimmt (Hypothese H3): Modelle bekommen ein deseasonalisiertes
    Signal und brauchen die Time-Features nicht mehr zu lernen.
  - Variante A ist der Gegenpol: Modelle muessen die volle Variabilitaet
    aus den Features (cyclic, Flags, Rolling) lernen.

Beide Outputs enthalten zusaetzlich die Metadaten-Spalten aus Stage 2
(was_flagged, interpolated) als Convenience fuer die spaetere
Trainings-Daten-Filterung in Stage 6. Diese sind ausdruecklich KEIN
Modell-Input - die Trennung muss in Stage 7 sauber gemacht werden.

Outputs:
    outputs/<dataset>/parquet/stage4_features_raw.parquet
    outputs/<dataset>/parquet/stage4_features_residual.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "demo_synthetic"

WMZ_NAMES = ["wmz_1", "wmz_2", "wmz_3"]

# Nacht-Definition: 22:00 - 05:59 (Stunden 22, 23, 0, 1, 2, 3, 4, 5)
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}

# Rolling-Windows in Stunden
ROLLING_SHORT_H = 6
ROLLING_LONG_H = 24


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


# ---------------------------------------------------------------------------
# Geteilte Feature-Builder
# ---------------------------------------------------------------------------

def build_time_features(index: pd.DatetimeIndex,
                        full: bool = True) -> pd.DataFrame:
    """Zyklische Encodings und binaere Zeit-Flags.

    full=True:  alle (cyclic + binaer + holiday + night)
    full=False: nur is_holiday (fuer Variante B - MSTL hat Zyklen bereits
                konsumiert; is_holiday ist irregulaer und nicht periodisch)
    """
    out = pd.DataFrame(index=index)

    # is_holiday immer (irregulaer, nicht von MSTL erfasst)
    years = list(range(index.year.min(), index.year.max() + 1))
    de_be_holidays = holidays.DE(subdiv="BE", years=years)
    out["is_holiday"] = pd.Series(
        [d.date() in de_be_holidays for d in index],
        index=index,
    ).astype("int8")

    if not full:
        return out

    hour = index.hour
    weekday = index.weekday   # Mo=0 ... So=6
    month = index.month

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["is_weekday"] = (weekday < 5).astype("int8")
    out["is_weekend"] = (weekday >= 5).astype("int8")
    out["is_night"] = pd.Series(
        [h in NIGHT_HOURS for h in hour], index=index
    ).astype("int8")

    return out


def build_rolling_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """5 rollierende Features fuer ein Signal: mean/std 6h+24h + deviation."""
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_rolling_mean_{ROLLING_SHORT_H}h"] = (
        series.rolling(ROLLING_SHORT_H, min_periods=ROLLING_SHORT_H // 2).mean()
    )
    out[f"{prefix}_rolling_mean_{ROLLING_LONG_H}h"] = (
        series.rolling(ROLLING_LONG_H, min_periods=ROLLING_LONG_H // 2).mean()
    )
    out[f"{prefix}_rolling_std_{ROLLING_SHORT_H}h"] = (
        series.rolling(ROLLING_SHORT_H, min_periods=ROLLING_SHORT_H // 2).std()
    )
    out[f"{prefix}_rolling_std_{ROLLING_LONG_H}h"] = (
        series.rolling(ROLLING_LONG_H, min_periods=ROLLING_LONG_H // 2).std()
    )
    out[f"{prefix}_deviation_{ROLLING_LONG_H}h"] = (
        series - out[f"{prefix}_rolling_mean_{ROLLING_LONG_H}h"]
    )
    return out


def build_variant_a(stage2: pd.DataFrame) -> pd.DataFrame:
    """Variante A: Rohdaten + volles Feature-Set."""
    parts: list[pd.DataFrame] = []

    # Pro Zaehler: Signal + Rolling-Stats
    for wmz in WMZ_NAMES:
        signal = stage2[[f"{wmz}_kw_mean"]].copy()
        rolling = build_rolling_features(stage2[f"{wmz}_kw_mean"], wmz)
        parts.append(pd.concat([signal, rolling], axis=1))

    # Geteilt: Zeit-Features (voll) + Wetter
    time_features = build_time_features(stage2.index, full=True)
    weather = stage2[["temperature", "humidity"]]
    parts.append(time_features)
    parts.append(weather)

    # Metadaten (fuer Stage-6-Filterung, KEIN Modell-Input)
    meta_cols = ([f"{wmz}_was_flagged" for wmz in WMZ_NAMES]
                 + [f"{wmz}_interpolated" for wmz in WMZ_NAMES])
    parts.append(stage2[meta_cols])

    out = pd.concat(parts, axis=1)
    out.index.name = "timestamp"
    return out


def build_variant_b(stage2: pd.DataFrame,
                    stage3: pd.DataFrame) -> pd.DataFrame:
    """Variante B: MSTL-Residuum + minimaler Kontext."""
    parts: list[pd.DataFrame] = []

    for wmz in WMZ_NAMES:
        residual = stage3[[f"{wmz}_residual"]].copy()
        trend = stage3[[f"{wmz}_trend"]]
        # rolling_std_24h NUR auf dem Residuum (Plateau-Detektor-Feature
        # gemaess H7 - Variance-Verlust signalisiert Plateau)
        rstd = pd.DataFrame({
            f"{wmz}_residual_rolling_std_{ROLLING_LONG_H}h":
                stage3[f"{wmz}_residual"]
                .rolling(ROLLING_LONG_H, min_periods=ROLLING_LONG_H // 2)
                .std()
        }, index=stage3.index)
        parts.append(pd.concat([residual, trend, rstd], axis=1))

    # Minimaler geteilter Kontext: is_holiday + temperature
    time_minimal = build_time_features(stage3.index, full=False)
    weather = stage2[["temperature"]]   # humidity weggelassen (schwache Korr.)
    parts.append(time_minimal)
    parts.append(weather)

    # Metadaten
    meta_cols = ([f"{wmz}_was_flagged" for wmz in WMZ_NAMES]
                 + [f"{wmz}_interpolated" for wmz in WMZ_NAMES])
    parts.append(stage2[meta_cols])

    out = pd.concat(parts, axis=1)
    out.index.name = "timestamp"
    return out


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------

def main() -> None:
    """Stage 4 ausfuehren: Merkmalstabellen je Detektions-Schiene erzeugen.

    Schreibt eine Tabelle fuer Schiene A (Rohsignal plus rollierende
    Statistiken) und eine fuer Schiene B (MSTL-Residuum, gleiche Statistiken).
    Schiene C braucht keine eigene Tabelle - sie arbeitet direkt auf dem
    Trend aus Stage 3.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    stage2_path = out_root / "parquet" / "stage2_hourly.parquet"
    stage3_path = out_root / "parquet" / "stage3_stl.parquet"
    out_a_path = out_root / "parquet" / "stage4_features_raw.parquet"
    out_b_path = out_root / "parquet" / "stage4_features_residual.parquet"

    if not stage2_path.is_file():
        raise SystemExit(f"Stage-2 nicht gefunden: {stage2_path}")
    if not stage3_path.is_file():
        raise SystemExit(f"Stage-3 nicht gefunden: {stage3_path}")

    print(f"Dataset: {args.dataset}")
    stage2 = pd.read_parquet(stage2_path)
    stage3 = pd.read_parquet(stage3_path)
    print(f"  Stage 2 input: {stage2.shape}")
    print(f"  Stage 3 input: {stage3.shape}")

    print("\nBaue Variante A (Rohdaten + volle Features)...")
    variant_a = build_variant_a(stage2)
    print(f"  Shape: {variant_a.shape}")

    print("Baue Variante B (Residuum + minimaler Kontext)...")
    variant_b = build_variant_b(stage2, stage3)
    print(f"  Shape: {variant_b.shape}")

    # --- Quality-Report -----------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 4 Quality Report")
    print("=" * 70)

    print("\n  Variante A Spalten (gruppiert):")
    sig_cols = [c for c in variant_a.columns if c.endswith("_kw_mean")]
    rolling_cols = [c for c in variant_a.columns if "_rolling_" in c or "_deviation_" in c]
    time_cols = [c for c in variant_a.columns
                 if c.endswith("_sin") or c.endswith("_cos")
                 or c.startswith("is_")]
    weather_cols = ["temperature", "humidity"]
    meta_cols = [c for c in variant_a.columns
                 if c.endswith("_was_flagged") or c.endswith("_interpolated")]
    print(f"    Signal-Spalten     : {len(sig_cols):>2}  {sig_cols}")
    print(f"    Rolling/Deviation  : {len(rolling_cols):>2}  ({ROLLING_SHORT_H}h+{ROLLING_LONG_H}h pro Zaehler)")
    print(f"    Zeit-Features      : {len(time_cols):>2}  {time_cols}")
    print(f"    Wetter             : {len(weather_cols):>2}  {weather_cols}")
    print(f"    Metadaten (kein Input): {len(meta_cols):>2}")

    print("\n  Variante B Spalten (gruppiert):")
    b_signal = [c for c in variant_b.columns if c.endswith("_residual")]
    b_context = [c for c in variant_b.columns if c.endswith("_trend")]
    b_rolling = [c for c in variant_b.columns if "_rolling_std_" in c]
    b_time = [c for c in variant_b.columns
              if c.endswith("_sin") or c.endswith("_cos")
              or c.startswith("is_")]
    b_weather = [c for c in variant_b.columns if c == "temperature"]
    print(f"    Residuum (Signal)  : {len(b_signal):>2}  {b_signal}")
    print(f"    Trend (Kontext)    : {len(b_context):>2}  {b_context}")
    print(f"    Rolling-Std auf Res: {len(b_rolling):>2}")
    print(f"    Zeit-Features      : {len(b_time):>2}  {b_time}")
    print(f"    Wetter             : {len(b_weather):>2}  {b_weather}")

    print("\n  is_holiday-Anteil (alle Jahre):")
    n_holiday_a = int(variant_a["is_holiday"].sum())
    pct = 100 * n_holiday_a / len(variant_a)
    print(f"    {n_holiday_a} Stunden ({pct:.2f} %) - "
          f"entspricht ca. {n_holiday_a // 24} Feiertagen")

    print("\n  NaN-Anteile (Signal-Spalten, Variante A):")
    for c in sig_cols:
        n = int(variant_a[c].isna().sum())
        print(f"    {c:<22} {n:>5}  ({100*n/len(variant_a):5.2f} %)")

    print("\n  NaN-Anteile (Signal-Spalten, Variante B):")
    for c in b_signal:
        n = int(variant_b[c].isna().sum())
        print(f"    {c:<22} {n:>5}  ({100*n/len(variant_b):5.2f} %)")

    # --- Schreiben ----------------------------------------------------------
    out_a_path.parent.mkdir(parents=True, exist_ok=True)
    variant_a.to_parquet(out_a_path)
    variant_b.to_parquet(out_b_path)
    print(f"\n  Wrote {out_a_path}")
    print(f"  Wrote {out_b_path}")


if __name__ == "__main__":
    main()
