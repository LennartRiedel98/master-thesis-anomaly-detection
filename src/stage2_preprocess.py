"""Stage 2 - Vorverarbeitung mit Dual-Channel-Fehlerfilter.

Liest ``stage1_raw_merged.parquet`` (minutenweise kumuliertes MWh + momentanes
kW + Wetter) und erzeugt eine bereinigte stundenbasierte Tabelle. Primaeres
Signal ist ``kW`` (mean pro Stunde) statt der frueher verwendeten kWh-Deltas.
Fehler werden auf beiden Kanaelen (MWh und kW) separat detektiert; geflaggte
Zeitstempel werden im kW-Signal genullt und kurze Luecken interpoliert.

Pipeline (entsprechend Methodik-Sektion 8.1):

  Stufe 1 - Fehlererkennung (minutenweise, pro Signal):
    Auf MWh (kumuliert):
      * is_mwh_reset    - kumulierter Wert faellt unter eigenes Maximum
      * is_mwh_spike    - unplausibler Sprung nach oben (Median-basiert)
      * is_mwh_plateau  - Standardabweichung approximativ 0 ueber Fenster
    Auf kW (momentan):
      * is_kw_spike     - unplausibler Sprung nach oben in kW
      * is_kw_plateau   - Standardabweichung approximativ 0 ueber Fenster
    Auf beiden gemeinsam:
      * is_gap          - beide Signale gleichzeitig NaN

  Stufe 2 - Bereinigung des kW-Signals:
    * Geflaggte Zeitstempel -> NaN in kW
    * Kurzluecken (<= 3 h) im kW-Signal linear interpoliert
    * Stuendliches Resampling: mean(kW)
    * Flags werden pro Stunde aggregiert (Anzahl Flag-Minuten je Typ
      sowie summarisches was_flagged-Boolean)

Plateau-Detektion ist cross-channel-aware: ein Sensor-Plateau wird nur
geflaggt, wenn der jeweils ANDERE Kanal in derselben Zeitspanne
Aktivitaet zeigt (sonst ist Konstanz das Resultat eines echten
Stillstands, kein Sensor-Fehler). Damit werden Sommer-Off-Phasen nicht
flaeschlich als Plateau geflaggt.

Zusaetzlich zur User-Spezifikation wird ``is_kw_negative`` geflaggt
(physikalisch unmoegliche Werte < 0 in kW). Detektion und Bereinigung
laufen nur auf WMZ-relevanten Zeilen (nicht auf Weather-only-Zeilen aus
dem Outer-Join in Stage 1).

Outputs:
    outputs/<dataset>/parquet/stage2_hourly.parquet
    outputs/<dataset>/csv/stage2_hourly.csv
    outputs/<dataset>/reports/stage2_flag_log.csv
    outputs/<dataset>/reports/stage2_gap_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pfade und Konstanten
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "gebaeude_a"

WMZ_NAMES = ["wmz_1", "wmz_2", "wmz_3"]
WEATHER_COLS = ["temperature", "humidity"]

WINDOW_START = pd.Timestamp("2019-11-19 00:00:00")
WINDOW_END = pd.Timestamp("2023-11-19 00:00:00")  # exklusiv

# Detection thresholds. Begruendungen siehe Methodik 8.1.
MWH_DROP_THRESHOLD = 1.0           # MWh - Reset wenn Wert mehr als 1 MWh
                                    # unter eigenem Maximum
MWH_SPIKE_THRESHOLD = 1.0          # MWh ueber lokalem Median = Spike
KW_SPIKE_FACTOR_OVER_MEDIAN = 10.0  # kW > 10x lokaler Median + Sockel = Spike
KW_SPIKE_SOCKET = 5.0              # kW Sockel, damit Median nahe 0 nicht
                                    # jeden Anstieg flaggt
PLATEAU_WINDOW_KW_MIN = 6          # rollendes Fenster fuer kW-Plateau-Pruefung
PLATEAU_WINDOW_MWH_MIN = 60        # MWh hat nur 3 Nachkommastellen (0.001 MWh
                                    # = 1 kWh) - 60 min Fenster gibt der
                                    # Mess-Praezision genug Raum, kleine Aende-
                                    # rungen ueberhaupt zu zeigen
PLATEAU_STD_THRESHOLD_MWH = 1e-4   # MWh - quasi 0
PLATEAU_STD_THRESHOLD_KW = 1e-2    # kW - quasi 0
# Cross-channel Aktivitaets-Schwellen fuer Plateau-Detektion:
PLATEAU_KW_ACTIVITY_THRESHOLD = 1.0    # kW > 1 = es liegt Aktivitaet vor
PLATEAU_MWH_DELTA_THRESHOLD = 1e-3     # MWh - kumulierte Aenderung im Fenster
ROLLING_MEDIAN_WINDOW_MIN = 201    # fuer Spike-Detektion via Median

MAX_INTERPOLATION_GAP_MIN = 3 * 60  # 3 Stunden in Minuten


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen.

    Neben der Datensatz-Wahl liegen hier die Schwellwerte des
    Zwei-Kanal-Fehlerfilters; sie sind bewusst ueber die Kommandozeile
    verstellbar, damit sich ein neuer Datensatz nachjustieren laesst, ohne
    den Code anzufassen.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Name des Unterordners unter outputs/. "
                        f"Default: {DEFAULT_DATASET}")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fehler-Detektoren (Stufe 1)
# ---------------------------------------------------------------------------

def detect_mwh_errors(mwh: pd.Series, kw: pd.Series) -> pd.DataFrame:
    """Drei Fehler-Typen auf einer kumulierten MWh-Serie erkennen.

    Reihenfolge ist wichtig: Spike-up wird zuerst erkannt und maskiert,
    bevor der Reset-Detektor sein laufendes Maximum berechnet. Sonst wuerde
    ein einzelner Aufwaerts-Glitch das Maximum vergiften und den Rest der
    Serie faelschlich als Reset markieren.

    Das Plateau ist cross-channel-aware: feuert nur, wenn der kW-Kanal in
    derselben Zeitspanne Aktivitaet (mean > Schwelle) zeigt. Damit werden
    echte Off-Phasen (Sommer-Stillstand) nicht als Sensor-Plateau fehl-
    interpretiert.
    """
    out = pd.DataFrame(index=mwh.index, dtype=bool)

    # Spike-up via lokalen Median
    local_median = (mwh.rolling(window=ROLLING_MEDIAN_WINDOW_MIN,
                                center=True, min_periods=5)
                       .median())
    is_spike = mwh > (local_median + MWH_SPIKE_THRESHOLD)
    out["is_mwh_spike"] = is_spike.fillna(False)

    # Reset auf der spike-bereinigten Serie
    mwh_clean = mwh.mask(is_spike)
    running_max = mwh_clean.cummax()
    is_reset = mwh < (running_max - MWH_DROP_THRESHOLD)
    out["is_mwh_reset"] = is_reset.fillna(False)

    # Plateau: MWh-Zuwachs im Fenster ist viel kleiner als die kW-basierte
    # Erwartung (= Sensor stuck waehrend Heizung laeuft).
    # 60-min-Fenster gibt der 3-stelligen MWh-Praezision Raum.
    # Wichtig: nur positive Zuwaechse betrachten - negative Zuwaechse sind
    # Reset-Artefakte, die bereits separat als is_mwh_reset geflaggt sind.
    kw_mean_window = kw.rolling(window=PLATEAU_WINDOW_MWH_MIN,
                                min_periods=PLATEAU_WINDOW_MWH_MIN).mean()
    expected_mwh_delta = kw_mean_window * (PLATEAU_WINDOW_MWH_MIN / 60.0) / 1000.0
    actual_mwh_delta = mwh.diff(PLATEAU_WINDOW_MWH_MIN)
    is_plateau = ((actual_mwh_delta >= 0)
                  & (actual_mwh_delta < expected_mwh_delta * 0.1)
                  & (kw_mean_window > PLATEAU_KW_ACTIVITY_THRESHOLD))
    out["is_mwh_plateau"] = is_plateau.fillna(False)

    return out


def detect_kw_errors(kw: pd.Series, mwh: pd.Series) -> pd.DataFrame:
    """Drei Fehler-Typen auf einer momentanen kW-Serie erkennen.

    Erweiterung gegenueber User-Spec: zusaetzlich ``is_kw_negative`` fuer
    physikalisch unmoegliche Werte < 0. Plateau ist cross-channel-aware:
    feuert nur, wenn die kumulierte MWh-Serie in derselben Zeitspanne
    angestiegen ist (= echte Heiz-Aktivitaet).
    """
    out = pd.DataFrame(index=kw.index, dtype=bool)

    # Negative kW (Sensor-Fehler, physikalisch unmoeglich)
    is_negative = kw < 0
    out["is_kw_negative"] = is_negative.fillna(False)

    # Spike-up: doppelte Bedingung - sowohl ueber dem lokalen Median als
    # auch ueber einer globalen Plausibilitaets-Schwelle (3 x das 95%-
    # Quantil aller Werte). Verhindert False Positives bei wmz_3, wo der
    # Sommer-Median gegen 0 geht und jeder Winter-Anstieg sonst flagt.
    local_median = (kw.rolling(window=ROLLING_MEDIAN_WINDOW_MIN,
                               center=True, min_periods=5)
                      .median())
    global_p95 = kw.quantile(0.95)
    relative_threshold = local_median * KW_SPIKE_FACTOR_OVER_MEDIAN + KW_SPIKE_SOCKET
    absolute_threshold = global_p95 * 3.0 + KW_SPIKE_SOCKET
    is_spike = (kw > relative_threshold) & (kw > absolute_threshold)
    out["is_kw_spike"] = is_spike.fillna(False)

    # Plateau: kW-Std approx 0 UND MWh ist im Fenster gestiegen
    rolling_std = kw.rolling(window=PLATEAU_WINDOW_KW_MIN, min_periods=PLATEAU_WINDOW_KW_MIN).std()
    mwh_delta_window = mwh.diff(PLATEAU_WINDOW_KW_MIN)   # MWh-Aenderung ueber Fenster
    is_plateau = ((rolling_std < PLATEAU_STD_THRESHOLD_KW)
                  & (mwh_delta_window > PLATEAU_MWH_DELTA_THRESHOLD))
    out["is_kw_plateau"] = is_plateau.fillna(False)

    return out


def detect_gap(mwh: pd.Series, kw: pd.Series) -> pd.Series:
    """Gap = beide Signale gleichzeitig NaN am selben Timestamp.

    Wird nur auf WMZ-relevanten Zeilen ausgewertet, sodass Weather-only-
    Zeilen aus dem Outer-Join in Stage 1 nicht faelschlich als Lueck-
    Eintraege gezaehlt werden.
    """
    return mwh.isna() & kw.isna()


# ---------------------------------------------------------------------------
# Interpolation und Aggregation
# ---------------------------------------------------------------------------

def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Lineare Interpolation nur fuer NaN-Runs <= max_gap (in Index-Schritten)."""
    is_nan = series.isna()
    if not is_nan.any():
        return series

    run_id = (is_nan != is_nan.shift()).cumsum()
    run_lengths = is_nan.groupby(run_id).transform("sum")
    fillable = is_nan & (run_lengths <= max_gap)

    interpolated = series.interpolate(method="linear", limit_direction="both")
    return series.where(~fillable, interpolated)


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------

def main() -> None:
    """Stage 2 ausfuehren: Minutenwerte pruefen, bereinigen, stuendlich mitteln.

    Ablauf: Roh-Minutenreihe aus Stage 1 laden, auf beiden Kanaelen getrennt
    nach Fehlern suchen (kumuliertes MWh und momentanes kW), auffaellige
    Zeitstempel im kW-Signal nullen, kurze Luecken interpolieren und
    anschliessend auf Stundenmittel verdichten. Die gesetzten Flags bleiben
    als eigene Spalten erhalten - sie sind spaeter Trainingsfilter und
    Ausschlusskriterium der Bewertung, aber nie Modell-Eingang.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    stage1_path = out_root / "parquet" / "stage1_raw_merged.parquet"
    out_parquet_dir = out_root / "parquet"
    out_csv_dir = out_root / "csv"
    out_report_dir = out_root / "reports"
    out_parquet_path = out_parquet_dir / "stage2_hourly.parquet"
    out_csv_path = out_csv_dir / "stage2_hourly.csv"
    out_flag_log_path = out_report_dir / "stage2_flag_log.csv"
    out_gap_log_path = out_report_dir / "stage2_gap_log.csv"

    if not stage1_path.is_file():
        raise SystemExit(
            f"Stage-1 output nicht gefunden: {stage1_path}\n"
            f"Lauf: python src/stage1_load.py --dataset {args.dataset}"
        )

    print(f"Dataset: {args.dataset}")
    print(f"Loading {stage1_path}")
    df = pd.read_parquet(stage1_path)
    print(f"  Stage 1 input: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Precipitation droppen (nicht im Feature-Plan)
    if "precipitation" in df.columns:
        df = df.drop(columns=["precipitation"])
        print("  precipitation Spalte entfernt")

    # WMZ-relevante Zeilen identifizieren. Weather-only-Zeilen aus dem
    # Outer-Join in Stage 1 enthalten WMZ-NaN, was kein Daten-Ausfall ist.
    wmz_value_cols = ([f"{w}_mwh" for w in WMZ_NAMES]
                      + [f"{w}_kw" for w in WMZ_NAMES])
    is_wmz_row = df[wmz_value_cols].notna().any(axis=1)
    wmz_df = df.loc[is_wmz_row, wmz_value_cols]
    print(f"  WMZ-relevante Zeilen: {len(wmz_df):,} von {len(df):,}")

    # --- Stufe 1: Fehler-Detektion pro Zaehler -----------------------------
    flag_dfs = []
    cleaned_kw_per_meter: dict[str, tuple[pd.Series, pd.Series]] = {}

    print("\nStufe 1 - Fehler-Detektion (minutenweise, auf WMZ-Zeilen):")
    for wmz in WMZ_NAMES:
        mwh = wmz_df[f"{wmz}_mwh"]
        kw = wmz_df[f"{wmz}_kw"]

        mwh_flags = detect_mwh_errors(mwh, kw)
        kw_flags = detect_kw_errors(kw, mwh)
        gap = detect_gap(mwh, kw)

        flags = pd.concat([mwh_flags, kw_flags], axis=1)
        flags["is_gap"] = gap

        # Stufe 2 (auf Minuten-Ebene): geflaggte Punkte in kW nullen
        any_flag = flags.any(axis=1)
        cleaned_kw = kw.where(~any_flag, np.nan)

        # Kurzluecken (<= 3 h = 180 Minuten) interpolieren
        interp_kw = interpolate_short_gaps(cleaned_kw, MAX_INTERPOLATION_GAP_MIN)
        was_interpolated = cleaned_kw.isna() & interp_kw.notna()
        cleaned_kw_per_meter[wmz] = (interp_kw, was_interpolated)

        flag_counts = flags.sum()
        print(f"  {wmz}: "
              f"reset={int(flag_counts['is_mwh_reset']):>6}  "
              f"mwh_spike={int(flag_counts['is_mwh_spike']):>5}  "
              f"mwh_plateau={int(flag_counts['is_mwh_plateau']):>6}  "
              f"kw_neg={int(flag_counts['is_kw_negative']):>5}  "
              f"kw_spike={int(flag_counts['is_kw_spike']):>5}  "
              f"kw_plateau={int(flag_counts['is_kw_plateau']):>6}  "
              f"gap={int(flag_counts['is_gap']):>4}")

        # Spaltennamen pro Zaehler praefixieren fuer das Zusammenfuegen
        flags.columns = [f"{wmz}_{c}" for c in flags.columns]
        flag_dfs.append(flags)

    minute_flags = pd.concat(flag_dfs, axis=1)

    # --- Auf stuendliches Raster aggregieren -------------------------------
    canonical_index = pd.date_range(WINDOW_START, WINDOW_END,
                                    freq="1h", inclusive="left")
    print(f"\nKanonischer Stundenindex: {len(canonical_index)} Stunden")

    # kW mean pro Stunde
    hourly_kw = pd.DataFrame(index=canonical_index)
    for wmz, (interp_kw, was_interpolated) in cleaned_kw_per_meter.items():
        hourly_kw[f"{wmz}_kw_mean"] = (
            interp_kw.resample("1h", label="left", closed="left").mean()
                     .reindex(canonical_index)
        )
        # Interpolations-Flag: irgendeine Minute in dieser Stunde wurde interpoliert
        hourly_kw[f"{wmz}_interpolated"] = (
            was_interpolated.resample("1h", label="left", closed="left").sum()
                            .reindex(canonical_index, fill_value=0) > 0
        )

    # Flag-Counts pro Stunde
    hourly_flags = pd.DataFrame(index=canonical_index)
    for col in minute_flags.columns:
        hourly_flags[f"{col}_count"] = (
            minute_flags[col].astype(int)
                            .resample("1h", label="left", closed="left").sum()
                            .reindex(canonical_index, fill_value=0)
                            .astype("int32")
        )

    # Pro-Zaehler-Summary: was_flagged (irgendein Flag in irgendeiner Minute)
    for wmz in WMZ_NAMES:
        wmz_flag_cols = [c for c in hourly_flags.columns
                         if c.startswith(f"{wmz}_") and c.endswith("_count")]
        hourly_flags[f"{wmz}_was_flagged"] = (
            hourly_flags[wmz_flag_cols].sum(axis=1) > 0
        )

    # Wetter
    hourly_weather = (df[WEATHER_COLS]
                      .resample("1h", label="left", closed="left")
                      .mean()
                      .reindex(canonical_index))

    out = pd.concat([hourly_kw, hourly_weather, hourly_flags], axis=1)
    out.index.name = "timestamp"

    # --- Berichte ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 2 v2 Quality Report")
    print("=" * 70)
    print(f"  Output shape    : {out.shape}")
    print(f"  Index span      : {out.index.min()} -> {out.index.max()}")

    print("\n  NaN counts (Wertspalten):")
    value_cols = [f"{w}_kw_mean" for w in WMZ_NAMES] + WEATHER_COLS
    for col in value_cols:
        n = int(out[col].isna().sum())
        pct = 100 * n / len(out)
        print(f"    {col:<22} {n:>6}   ({pct:5.2f} %)")

    print("\n  Pro-Zaehler Flag-Summary (Stunden mit irgendeinem Flag):")
    for wmz in WMZ_NAMES:
        n_flagged = int(out[f"{wmz}_was_flagged"].sum())
        n_interpolated = int(out[f"{wmz}_interpolated"].sum())
        print(f"    {wmz}: was_flagged={n_flagged:>5} ({100*n_flagged/len(out):.1f} %)  "
              f"interpolated={n_interpolated:>4}")

    print("\n  kW-Mean Sanity-Werte (ignoring NaN):")
    for wmz in WMZ_NAMES:
        s = out[f"{wmz}_kw_mean"].dropna()
        if len(s) == 0:
            print(f"    {wmz}_kw_mean: ALL NaN!")
            continue
        print(f"    {wmz}_kw_mean: "
              f"min={s.min():>8.2f}  "
              f"median={s.median():>8.2f}  "
              f"p99={s.quantile(0.99):>8.2f}  "
              f"max={s.max():>9.2f}")

    # Flag-Log-Tabelle: pro Stunde aggregierte Counts (kompakt)
    flag_log_cols = [c for c in out.columns if c.endswith("_count")]
    flag_log = out[flag_log_cols].copy()
    flag_log = flag_log.loc[(flag_log > 0).any(axis=1)]  # nur Stunden mit irgendeinem Flag

    # Gap-Log: nur Reihen, in denen es Gaps gab
    gap_cols = [c for c in flag_log_cols if "_is_gap_count" in c]
    gap_log_rows = flag_log[gap_cols]
    gap_log = gap_log_rows.loc[(gap_log_rows > 0).any(axis=1)]

    # --- Outputs schreiben -------------------------------------------------
    out_parquet_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_report_dir.mkdir(parents=True, exist_ok=True)

    out.to_parquet(out_parquet_path)
    out.to_csv(out_csv_path)
    flag_log.to_csv(out_flag_log_path)
    gap_log.to_csv(out_gap_log_path)

    print(f"\n  Wrote {out_parquet_path}")
    print(f"  Wrote {out_csv_path}")
    print(f"  Wrote {out_flag_log_path}")
    print(f"  Wrote {out_gap_log_path}")


if __name__ == "__main__":
    main()
