"""Stage 1 - WMZ-Waermemengen-CSVs mit Open-Meteo-Wetterdaten zusammenfuehren.

Liest die deutsch formatierten WMZ-CSVs (kumulierte MWh + momentane kW
je Zaehler, Abtastung rund im Minutentakt) sowie die stuendliche
Open-Meteo-CSV, legt beide ueber einen Outer-Join auf einen gemeinsamen
Zeitstempel-Index und schneidet auf das saubere Vier-Jahres-Fenster zu:

    [2019-11-19 00:00, 2023-11-19 00:00)   ->  letzte Stunde: 2023-11-18 23:00

Hier wird bewusst **nicht** resampelt, differenziert oder interpoliert - das
gehoert in Stage 2. Die Ausgabe behaelt die native Minutenaufloesung der WMZ;
die Wetterspalten sind auf allen nicht-stuendlichen Zeitstempeln NaN und
werden erst in Stage 2 ausgerichtet.

Ausgaben:
    outputs/<dataset>/parquet/stage1_raw_merged.parquet
    outputs/<dataset>/csv/stage1_raw_merged.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Pfade und Konstanten
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "demo_synthetic"

# Die Eingabedateien werden ueber ein Namensmuster gefunden, nicht ueber eine
# feste Liste: So laesst sich ein weiterer Datensatz allein durch Ablegen der
# Dateien unter data/<name>/ anschliessen (siehe README, "Eigenen Datensatz
# hinzufuegen"). Die Sortierung nach Dateiname ist nur Kosmetik fuer die
# Konsolenausgabe - zusammengefuehrt wird ohnehin ueber den Zeitstempel.
WMZ_GLOB = "logging_heat-energy_*.csv"
WEATHER_GLOB = "open-meteo-*.csv"

# Zaehler-ID -> wmz_N. Die Nummerierung folgt der Spaltenposition im CSV-Kopf:
# der erste im Kopf auftretende Zaehler wird wmz_1, der zweite wmz_2 usw. Damit
# haengt der Code nicht an konkreten Zaehlernummern.
#
# ACHTUNG (Methodik 3.1.1): Die Spaltenreihenfolge der Logging-Dateien ist
# gegenueber der Nummerierung des Waermeversorgers **vertauscht**. Die hier
# vergebenen Namen wmz_1/2/3 sind reine Spaltenbezeichner; physikalisch
# versorgen sie:
#     wmz_1 = Trinkwarmwasser Allgemeinbereiche          (Waermeuebergabe 3)
#     wmz_2 = Trinkwarmwasser + dynamische Heizung Kueche (Waermeuebergabe 2)
#     wmz_3 = Heizung Laborgebaeude + Fussbodenheizung Kueche (Waermeuebergabe 1)
# Die Zuordnung ist ueber die MSTL-Strength-Profile gegengeprueft (der
# Trinkwarmwasser-Zaehler zeigt die staerkste Tagesperiodik). Sie wird
# absichtlich **nicht** im Code korrigiert: Alle Ergebnisse, Abbildungen und
# Tabellen der Arbeit sind auf die Spaltennamen bezogen; eine Umbenennung
# wuerde jede Zahl unlesbar machen.


WINDOW_START = pd.Timestamp("2019-11-19 00:00:00")
WINDOW_END = pd.Timestamp("2023-11-19 00:00:00")  # exklusiv


def meter_map_from_header(columns: list[str]) -> dict[str, str]:
    """Zaehler-IDs in Kopfzeilen-Reihenfolge auf ``wmz_1``, ``wmz_2``, ... abbilden.

    Erwartet Spaltennamen der Form ``<Zaehler-ID> / <Einheit>``. Jede ID kommt
    zweimal vor (MWh und kW); gezaehlt wird ihr **erstes** Auftreten.
    """
    mapping: dict[str, str] = {}
    for col in columns:
        meter_id, sep, _unit = col.partition(" / ")
        if not sep:
            continue
        meter_id = meter_id.strip()
        if meter_id not in mapping:
            mapping[meter_id] = f"wmz_{len(mapping) + 1}"
    return mapping


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (nur die Datensatz-Wahl)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=("Name des Unterordners unter data/ (und outputs/). "
                         "Erwarteter Aufbau: data/<dataset>/ mit den "
                         "WMZ-CSVs und einer Open-Meteo-CSV. "
                         f"Standard: {DEFAULT_DATASET}"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Lade-Funktionen
# ---------------------------------------------------------------------------

def load_wmz_file(path: Path) -> pd.DataFrame:
    """Eine WMZ-CSV laden und nach Zeitstempel indiziert zurueckgeben.

    Die Dateien sind deutsch formatiert (Semikolon als Trenner, Komma als
    Dezimalzeichen) und tragen Datum und Uhrzeit in zwei getrennten Spalten.
    Die Messspalten heissen im Original ``<Zaehler-ID> / <Einheit>`` und
    werden hier auf das sprechende Schema ``wmz_<N>_<einheit>`` umbenannt.
    """
    df = pd.read_csv(path, sep=";", decimal=",")

    timestamp = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d.%m.%Y %H:%M:%S",
    )
    # Zeilenzaehler, das Datum/Uhrzeit-Paar und eine evtl. leere Endspalte weg.
    drop_cols = [c for c in df.columns
                 if c in ("No.", "Date", "Time") or c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    df.index = timestamp
    df.index.name = "timestamp"

    meter_map = meter_map_from_header(list(df.columns))
    rename = {}
    for col in df.columns:
        meter_id, _, unit = col.partition(" / ")
        wmz_name = meter_map[meter_id.strip()]
        rename[col] = f"{wmz_name}_{unit.strip().lower()}"
    df = df.rename(columns=rename)
    return df


def load_weather(path: Path) -> pd.DataFrame:
    """Stuendliche Open-Meteo-CSV laden.

    Die ersten drei Zeilen der Datei enthalten einen Metadaten-Block
    (Koordinaten, Hoehe, Zeitzone) und werden uebersprungen. Die Spalten
    werden auf kurze, einheitenfreie Namen gebracht, weil die Einheit in der
    ganzen Pipeline konstant bleibt.
    """
    df = pd.read_csv(path, skiprows=3, parse_dates=["time"])
    df = df.rename(columns={
        "time": "timestamp",
        "temperature_2m (°C)": "temperature",
        "relative_humidity_2m (%)": "humidity",
        "precipitation (mm)": "precipitation",
    }).set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------

def main() -> None:
    """Stage 1 ausfuehren: laden, zusammenfuehren, zuschneiden, schreiben.

    Reihenfolge: alle WMZ-Dateien einlesen und aneinanderhaengen,
    Duplikat-Zeitstempel an den Dateigrenzen entfernen (die Exporte
    ueberlappen sich um wenige Minuten), Wetter per Outer-Join anfuegen, auf
    das Vier-Jahres-Fenster zuschneiden und einen Qualitaetsbericht auf die
    Konsole schreiben. Der Bericht ist die erste Kontrolle, ob ein neuer
    Datensatz die erwartete Form hat.
    """
    args = parse_args()
    data_dir = ROOT / "data" / args.dataset
    out_parquet_dir = ROOT / "outputs" / args.dataset / "parquet"
    out_csv_dir = ROOT / "outputs" / args.dataset / "csv"
    out_parquet_path = out_parquet_dir / "stage1_raw_merged.parquet"
    out_csv_path = out_csv_dir / "stage1_raw_merged.csv"

    if not data_dir.is_dir():
        raise SystemExit(f"Datensatz-Ordner nicht gefunden: {data_dir}")

    print(f"Dataset: {args.dataset}")
    print(f"Lade aus {data_dir}\n")

    # --- WMZ-Dateien ---------------------------------------------------------
    wmz_paths = sorted(data_dir.glob(WMZ_GLOB))
    if not wmz_paths:
        raise SystemExit(
            f"Keine WMZ-Dateien ({WMZ_GLOB}) in {data_dir} gefunden. "
            "Fuer den mitgelieferten Demo-Datensatz zuerst "
            "'python src/tools/make_synthetic_data.py' ausfuehren."
        )

    print("WMZ-Dateien:")
    wmz_parts = []
    for path in wmz_paths:
        df = load_wmz_file(path)
        print(f"  {path.name:<55} {len(df):>8} Zeilen  "
              f"{df.index.min()} -> {df.index.max()}")
        wmz_parts.append(df)

    wmz = pd.concat(wmz_parts).sort_index()
    n_concat = len(wmz)
    wmz = wmz[~wmz.index.duplicated(keep="first")]
    print(f"\n  Zusammengehaengt: {n_concat} Zeilen -> nach Dedup "
          f"(erster Treffer gewinnt): {len(wmz)} Zeilen "
          f"({n_concat - len(wmz)} doppelte Zeitstempel entfernt)")

    # --- Wetter --------------------------------------------------------------
    weather_paths = sorted(data_dir.glob(WEATHER_GLOB))
    if len(weather_paths) != 1:
        raise SystemExit(
            f"Genau eine Wetterdatei ({WEATHER_GLOB}) erwartet, "
            f"{len(weather_paths)} gefunden in {data_dir}"
        )
    weather = load_weather(weather_paths[0])
    print(f"\nWetter: {len(weather)} Zeilen  "
          f"{weather.index.min()} -> {weather.index.max()}")

    # --- Zusammenfuehren (Outer-Join auf dem Zeitstempel) --------------------
    merged = wmz.join(weather, how="outer").sort_index()

    # --- Auf das saubere Fenster zuschneiden ---------------------------------
    n_before_clip = len(merged)
    mask = (merged.index >= WINDOW_START) & (merged.index < WINDOW_END)
    merged = merged.loc[mask]
    print(f"\nZugeschnitten auf [{WINDOW_START}, {WINDOW_END}): "
          f"{len(merged)} Zeilen  ({n_before_clip - len(merged)} entfernt)")

    # --- Qualitaetsbericht ---------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 1 - Qualitaetsbericht")
    print("=" * 70)
    print(f"  Form           : {merged.shape}")
    print(f"  Zeitraum       : {merged.index.min()} -> {merged.index.max()}")
    print(f"  Spalten        : {list(merged.columns)}")
    print("\n  Fehlwerte je Spalte:")
    for col, n in merged.isna().sum().items():
        pct = 100 * n / len(merged)
        print(f"    {col:<25} {n:>9}   ({pct:5.2f} %)")

    # --- Ausgaben schreiben --------------------------------------------------
    out_parquet_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_parquet_path)
    merged.to_csv(out_csv_path)
    print(f"\nGeschrieben: {out_parquet_path}")
    print(f"Geschrieben: {out_csv_path}")


if __name__ == "__main__":
    main()
