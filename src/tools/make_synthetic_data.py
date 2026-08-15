"""Erzeugt den synthetischen Demo-Datensatz unter ``data/demo_synthetic/``.

Warum es dieses Skript gibt
---------------------------
Die Arbeit wurde auf **realen Betriebsdaten** eines Gewerbegebaeudes gerechnet.
Diese Daten sind vertraulich und daher **nicht Teil der Veroeffentlichung**
(siehe README, Abschnitt "Daten"). Damit die Pipeline trotzdem ohne Zugang zu
den Originaldaten lauffaehig, pruefbar und vorfuehrbar bleibt, erzeugt dieses
Skript einen **vollstaendig synthetischen Ersatzdatensatz im identischen
Dateiformat**.

Was uebernommen wird - und was nicht
------------------------------------
Nachgebildet ist allein die **Form** der Originaldateien, weil genau daran die
Lade- und Qualitaetssicherungsschritte haengen:

* fuenf Logging-CSVs mit Semikolon-Trenner, deutschem Dezimalkomma,
  CRLF-Zeilenenden und einer leeren Endspalte (die abschliessenden ``;``),
* Spaltenkoepfe der Form ``<Zaehler-ID> / MWh`` bzw. ``<Zaehler-ID> / kW``
  fuer drei Zaehler, mit den zaehlerspezifischen Nachkommastellen des
  Originals (3/1 bzw. 1/2),
* die Blockstruktur der Exporte: Der Zeilenzaehler ``No.`` springt an
  Loggerneustarts auf 1 zurueck, der Sekundenversatz des Minutenrasters
  wechselt dabei, und ein Neustart schreibt eine **Nullzeile** - genau die
  Reset-Artefakte, die Stage 2 protokolliert,
* die Zeitachse inklusive der realen Dateigrenzen, der mehrwoechigen Luecke im
  Spaetsommer 2020 und der fehlenden Stunde bei jeder Sommerzeitumstellung,
* eine stuendliche Open-Meteo-CSV mit dem dreizeiligen Metadatenkopf.

**Alle Messwerte sind erfunden.** Sie stammen aus dem unten stehenden Modell
mit festem Zufallsstartwert, nicht aus den Originaldaten; uebernommen wurden
weder Messwerte noch Zaehlernummern noch der Standort. Die Groessenordnungen
sind so gewaehlt, dass die Reihen physikalisch plausibel bleiben und die
Pipeline nicht in entartete Faelle laeuft.

Das Lastmodell
--------------
Grundlage ist eine synthetische Aussentemperatur (Jahresgang + Tagesgang +
AR(1)-Wetterrauschen) und daraus der Heizbedarf ``max(0, T_heiz - T)``. Die
drei Zaehler bekommen bewusst **verschiedene Strukturprofile**, damit die
MSTL-Zerlegung und die typspezifische Auswertung etwas zu unterscheiden haben:

* ``wmz_1`` - Trinkwarmwasser: schwach strukturiert, hoher Rauschanteil,
  nur lose an die Temperatur gekoppelt.
* ``wmz_2`` - Kueche: stark tages- und wochenperiodisch, werktags
  schubweise Spitzen, am Wochenende nahezu aus.
* ``wmz_3`` - Raumheizung: trenddominiert, fast vollstaendig
  temperaturgetrieben, im Sommer exakt null (Abschaltung), nachts abgesenkt.

Zusaetzlich sind zwei **Makroereignisse** eingeplant, damit
``src/tools/stage3_macroevents.py`` einen Gegenstand hat: ein Einbruch der
Kuechenlast im Fruehjahr 2020 und eine abgesenkte Heizlast ab Herbst 2022.
Beide sind hier gesetzt, nicht gemessen.

Aufruf
------
    python src/tools/make_synthetic_data.py
    python src/tools/make_synthetic_data.py --dataset demo_synthetic --seed 20260815

Der Lauf ist bei gleichem Startwert deterministisch: Er erzeugt bei jedem
Aufruf byte-identische Dateien (rund 140 MB, etwa eine Minute).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "demo_synthetic"
DEFAULT_SEED = 20260815

# --- Zeitachse -------------------------------------------------------------
# Beginn/Ende decken die Spannweite aller Logdateien ab; die Wetterdatei endet
# frueher, weil Stage 1 ohnehin auf [2019-11-19, 2023-11-19) zuschneidet.
GRID_START = pd.Timestamp("2019-11-19 00:00:00")
GRID_END = pd.Timestamp("2024-01-11 00:00:00")
WEATHER_START = pd.Timestamp("2019-11-19 00:00:00")
WEATHER_END = pd.Timestamp("2023-11-19 23:00:00")

# --- Dateistruktur ---------------------------------------------------------
# Je Datei eine Liste von Bloecken (Beginn, Ende, Sekundenversatz). Ein Block
# ist ein durchgehender Loggerlauf; an jeder Blockgrenze startet ``No.`` neu.
# Die Grenzen entsprechen denen der Originalexporte - das ist Metadatum der
# Aufzeichnung, kein Messwert.
FILES: list[tuple[str, list[tuple[str, str, int]]]] = [
    ("logging_heat-energy_demo_2019.11-2020.08.csv", [
        ("2019-11-19 07:14:28", "2020-07-11 08:56:28", 28),
        ("2020-07-11 12:23:47", "2020-08-27 13:32:47", 47),
    ]),
    ("logging_heat-energy_demo_2020.09-2021.01.csv", [
        ("2020-09-10 08:36:47", "2021-01-07 06:13:47", 47),
    ]),
    ("logging_heat-energy_demo_2021.01-2021.08.csv", [
        ("2021-01-07 06:15:47", "2021-08-16 15:23:47", 47),
    ]),
    ("logging_heat-energy_demo_2021.08-2022.08.csv", [
        ("2021-08-16 15:29:47", "2021-11-30 12:18:47", 47),
        ("2021-11-30 12:41:09", "2022-08-31 09:27:09", 9),
    ]),
    ("logging_heat-energy_demo_2022.08-2024.01.csv", [
        ("2022-08-31 09:31:09", "2022-10-07 14:24:09", 9),
        ("2022-10-07 14:36:36", "2023-06-10 08:06:36", 36),
        ("2023-06-10 11:48:31", "2024-01-10 14:37:31", 31),
    ]),
]
WEATHER_FILENAME = "open-meteo-demo.csv"

# Frei gewaehlte Zaehlernummern ohne Bezug zu realen Geraeten. Die Reihenfolge
# im Dateikopf legt die Zuordnung zu wmz_1/2/3 fest (siehe stage1_load.py).
METER_IDS = ["10000001", "10000002", "10000003"]
# Nachkommastellen je Zaehler: (MWh, kW). Im Original protokolliert der dritte
# Zaehler groeber im Zaehlerstand und feiner in der Leistung.
METER_DECIMALS = [(3, 1), (3, 1), (1, 2)]
# Zaehlerstaende zu Beginn der Aufzeichnung (MWh), frei gewaehlt.
METER_OFFSETS = [200.0, 210.0, 380.0]

# --- Lastmodell ------------------------------------------------------------
T_MEAN = 10.0          # Jahresmittel der Aussentemperatur (Grad C)
T_YEAR_AMP = 9.5       # halbe Spannweite des Jahresgangs
T_DAY_AMP = 2.6        # halbe Spannweite des Tagesgangs
T_HEATING_LIMIT = 15.0  # Heizgrenze: darunter entsteht Heizbedarf

# Makroereignisse (gesetzt, nicht gemessen)
EVENT_SHUTDOWN = ("2020-03-16", "2020-06-15")   # Kuechenlast bricht ein
EVENT_SAVING = ("2022-09-01", "2023-04-15")     # Heizlast abgesenkt


# ---------------------------------------------------------------------------
# Zeitachse
# ---------------------------------------------------------------------------

def last_sunday_of_march(year: int) -> pd.Timestamp:
    """Datum der Sommerzeitumstellung im Fruehjahr (letzter Sonntag im Maerz)."""
    day = pd.Timestamp(year=year, month=3, day=31)
    while day.weekday() != 6:  # 6 == Sonntag
        day -= pd.Timedelta(days=1)
    return day


def drop_dst_gap(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Die bei der Zeitumstellung uebersprungene Stunde entfernen.

    Die Zeitstempel der Logdateien sind naive Ortszeit. In der Nacht zum
    letzten Maerzsonntag existiert 02:00-02:59 nicht; im Original zeigt sich
    das als Luecke von 1:01 h. Die Rueckstellung im Oktober erzeugt dagegen
    doppelte Zeitstempel, die Stage 1 beim Dedup verwirft - die bleiben hier
    bewusst unmodelliert, weil sie im Original nur teilweise auftreten.
    """
    keep = np.ones(len(index), dtype=bool)
    for year in range(int(index.year.min()), int(index.year.max()) + 1):
        start = last_sunday_of_march(year) + pd.Timedelta(hours=2)
        keep &= ~((index >= start) & (index < start + pd.Timedelta(hours=1)))
    return index[keep]


def block_index(start: str, end: str) -> pd.DatetimeIndex:
    """Minutenraster eines Loggerlaufs, ohne die Sommerzeitluecke."""
    return drop_dst_gap(pd.date_range(start, end, freq="1min"))


# ---------------------------------------------------------------------------
# Wetter
# ---------------------------------------------------------------------------

def make_weather(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Stuendliche Aussentemperatur, Luftfeuchte und Niederschlag erzeugen.

    Die Temperatur setzt sich aus Jahresgang, Tagesgang und einem
    AR(1)-Prozess zusammen. Der AR(1)-Anteil traegt die Wetterlage: Er sorgt
    fuer mehrtaegige Kaelte- und Waermeperioden statt fuer weisses Rauschen -
    ohne ihn haette die Temperaturreihe keine realistische Persistenz und die
    Kreuzkorrelationsanalyse nichts zu zeigen.
    """
    doy = index.dayofyear.to_numpy(dtype=float)
    hour = index.hour.to_numpy(dtype=float)

    # Jahresgang: Minimum Mitte Januar (doy 15), Maximum Mitte Juli.
    seasonal = -T_YEAR_AMP * np.cos(2 * np.pi * (doy - 15.0) / 365.25)
    # Tagesgang: Minimum vor Sonnenaufgang, Maximum am spaeten Nachmittag.
    daily = -T_DAY_AMP * np.cos(2 * np.pi * (hour - 15.0) / 24.0)

    # AR(1)-Wetterlage mit einer Persistenz von rund zwei Tagen.
    rho, sigma = 0.965, 1.55
    innovations = rng.normal(0.0, sigma, size=len(index))
    weather_state = np.empty(len(index))
    weather_state[0] = innovations[0]
    for i in range(1, len(index)):
        weather_state[i] = rho * weather_state[i - 1] + innovations[i]

    temperature = T_MEAN + seasonal + daily + weather_state

    # Luftfeuchte laeuft der Temperaturabweichung entgegen.
    anomaly = temperature - (T_MEAN + seasonal)
    humidity = np.clip(78.0 - 2.4 * anomaly + rng.normal(0.0, 6.0, len(index)),
                       28.0, 100.0)

    # Niederschlag: meist trocken, sonst gammaverteilte Mengen.
    wet = rng.random(len(index)) < 0.085
    precipitation = np.where(wet, rng.gamma(1.4, 0.55, len(index)), 0.0)

    return pd.DataFrame(
        {"temperature": temperature, "humidity": humidity,
         "precipitation": precipitation},
        index=index,
    )


# ---------------------------------------------------------------------------
# Lastprofile
# ---------------------------------------------------------------------------

def _smooth_window(hour: np.ndarray, start: float, end: float,
                   edge: float = 1.0) -> np.ndarray:
    """Weiche 0/1-Fensterfunktion ueber der Tageszeit.

    Harte Rechteckfenster wuerden in der MSTL-Zerlegung als Sprung erscheinen
    und das Residuum an den Flanken dominieren; die Flanken werden deshalb
    ueber ``edge`` Stunden linear aufgeblendet.
    """
    rise = np.clip((hour - start) / edge, 0.0, 1.0)
    fall = np.clip((end - hour) / edge, 0.0, 1.0)
    return np.minimum(rise, fall)


def make_loads(index: pd.DatetimeIndex, temperature: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Momentanleistung der drei Zaehler in kW erzeugen.

    Rueckgabe: Array der Form ``(len(index), 3)``.
    """
    hour = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    weekday = index.dayofweek.to_numpy()
    is_weekend = weekday >= 5
    heat_demand = np.clip(T_HEATING_LIMIT - temperature, 0.0, None)

    def correlated_noise(scale: float, rho: float = 0.98) -> np.ndarray:
        """Multiplikatives Rauschen mit kurzer Erinnerung (Minutenskala)."""
        raw = rng.normal(0.0, 1.0, size=len(index))
        smoothed = pd.Series(raw).ewm(alpha=1.0 - rho).mean().to_numpy(copy=True)
        # Die Glaettung senkt die Varianz; wieder auf Einheitsvarianz bringen.
        smoothed /= smoothed.std()
        return np.exp(scale * smoothed - 0.5 * scale ** 2)

    # --- wmz_1: Trinkwarmwasser, schwach strukturiert ----------------------
    dhw_profile = (1.0
                   + 0.30 * _smooth_window(hour, 6.0, 9.0, 1.5)
                   + 0.22 * _smooth_window(hour, 17.0, 21.0, 1.5))
    wmz_1 = (8.65 + 0.11 * heat_demand) * dhw_profile * correlated_noise(0.16)
    wmz_1 *= np.where(is_weekend, 0.88, 1.0)

    # --- wmz_2: Kueche, stark tages-/wochenperiodisch ----------------------
    # Die Grundlast traegt bewusst eigenes Rauschen: Eine exakt konstante
    # Grundlast waere ein Dauerplateau und wuerde den Stage-2-Filter in jeder
    # Nacht ausloesen - ein Modellartefakt, keine Sensorpathologie.
    kitchen = _smooth_window(hour, 6.0, 15.0, 1.2)
    kitchen = kitchen * np.where(is_weekend, 0.12, 1.0)
    wmz_2 = ((3.2 + 0.05 * heat_demand) * correlated_noise(0.30)
             + kitchen * (22.0 + 0.60 * heat_demand) * correlated_noise(0.35))

    # --- wmz_3: Raumheizung, trenddominiert --------------------------------
    setback = np.where((hour >= 22.0) | (hour < 5.0), 0.72, 1.0)
    wmz_3 = 15.0 * heat_demand * setback * correlated_noise(0.20)

    # --- Makroereignisse ---------------------------------------------------
    shutdown = (index >= EVENT_SHUTDOWN[0]) & (index < EVENT_SHUTDOWN[1])
    wmz_2 = np.where(shutdown, wmz_2 * 0.30, wmz_2)
    wmz_1 = np.where(shutdown, wmz_1 * 0.74, wmz_1)

    saving = (index >= EVENT_SAVING[0]) & (index < EVENT_SAVING[1])
    wmz_3 = np.where(saving, wmz_3 * 0.87, wmz_3)

    return np.column_stack([wmz_1, wmz_2, wmz_3])


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def format_column(values: np.ndarray, decimals: int) -> np.ndarray:
    """Float-Spalte im deutschen Zahlenformat als Zeichenkette."""
    text = np.char.mod(f"%.{decimals}f", values)
    return np.char.replace(text, ".", ",")


def write_meter_file(path: Path, blocks: list[tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]],
                     with_reset: list[bool]) -> int:
    """Eine Logging-CSV im Originalformat schreiben.

    ``blocks`` enthaelt je Loggerlauf den Zeitindex, die Zaehlerstaende (MWh)
    und die Momentanleistungen (kW). ``with_reset`` legt fest, welcher Block
    mit einer Nullzeile beginnt - das Reset-Artefakt eines Loggerneustarts.
    """
    n_rows = 0
    for position, ((idx, mwh, kw), reset) in enumerate(zip(blocks, with_reset)):
        frame = pd.DataFrame({
            "No.": np.arange(1, len(idx) + 1),
            "Date": idx.strftime("%d.%m.%Y"),
            "Time": idx.strftime("%H:%M:%S"),
        })
        for j, meter_id in enumerate(METER_IDS):
            dec_mwh, dec_kw = METER_DECIMALS[j]
            frame[f"{meter_id} / MWh"] = format_column(mwh[:, j], dec_mwh)
            frame[f"{meter_id} / kW"] = format_column(kw[:, j], dec_kw)

        if reset:
            # Nullzeile: Der Logger meldet nach einem Neustart erst einmal 0.
            for j, meter_id in enumerate(METER_IDS):
                dec_mwh, dec_kw = METER_DECIMALS[j]
                frame.loc[0, f"{meter_id} / MWh"] = format_column(np.zeros(1), dec_mwh)[0]
                frame.loc[0, f"{meter_id} / kW"] = format_column(np.zeros(1), dec_kw)[0]

        # Leere Endspalte: erzeugt das abschliessende ";" jeder Zeile und den
        # ueberzaehligen Trenner im Kopf - genau wie im Originalexport.
        frame[""] = ""
        frame.to_csv(path, sep=";", index=False, lineterminator="\r\n",
                     encoding="utf-8", mode="w" if position == 0 else "a",
                     header=position == 0)
        n_rows += len(idx)

    return n_rows


def write_weather_file(path: Path, weather: pd.DataFrame) -> int:
    """Open-Meteo-CSV mit dem dreizeiligen Metadatenkopf schreiben.

    Die Koordinaten sind bewusst grob auf die Stadtmitte gesetzt: Der reale
    Standort gehoert nicht in die veroeffentlichte Fassung, das Format der
    Kopfzeilen aber schon - Stage 1 ueberspringt genau drei Zeilen.
    """
    frame = weather.loc[WEATHER_START:WEATHER_END]
    lines = [
        "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation",
        "52.52,13.405,34.0,7200,Europe/Berlin,GMT+2",
        "",
        "time,temperature_2m (°C),relative_humidity_2m (%),precipitation (mm)",
    ]
    stamps = frame.index.strftime("%Y-%m-%dT%H:%M").to_numpy()
    for stamp, temp, hum, prec in zip(stamps, frame["temperature"].to_numpy(),
                                      frame["humidity"].to_numpy(),
                                      frame["precipitation"].to_numpy()):
        lines.append(f"{stamp},{temp:.1f},{hum:.0f},{prec:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return len(frame)


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Zielordner unter data/. Standard: {DEFAULT_DATASET}")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Zufallsstartwert. Standard: {DEFAULT_SEED}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / "data" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ziel : {out_dir}")
    print(f"Seed : {args.seed}\n")

    # --- Wetter auf dem Stundenraster --------------------------------------
    hourly = pd.date_range(GRID_START, GRID_END, freq="1h")
    weather = make_weather(hourly, rng)

    # --- Last auf einem durchgehenden Minutenraster ------------------------
    # Durchgehend, damit die Zaehlerstaende auch ueber Aufzeichnungsluecken
    # hinweg korrekt weiterlaufen: Ein realer Zaehler zaehlt weiter, auch wenn
    # der Logger nichts schreibt.
    minutes = pd.date_range(GRID_START, GRID_END, freq="1min")
    x_hour = hourly.view("int64").astype(np.float64)
    x_min = minutes.view("int64").astype(np.float64)
    temp_min = np.interp(x_min, x_hour, weather["temperature"].to_numpy())

    kw_grid = make_loads(minutes, temp_min, rng)
    # Zaehlerstand = Integral der Leistung; 1 Minute bei kW ergibt kWh/60.
    mwh_grid = METER_OFFSETS + np.cumsum(kw_grid, axis=0) / 60.0 / 1000.0

    print("Erzeugte Reihen (kW):")
    for j in range(3):
        col = kw_grid[:, j]
        print(f"  wmz_{j + 1}:  Mittel {col.mean():7.2f}   Max {col.max():8.2f}   "
              f"Nullanteil {100 * (col == 0).mean():5.1f} %")
    print()

    # --- Dateien schreiben --------------------------------------------------
    total_rows = 0
    for file_no, (fname, block_spec) in enumerate(FILES):
        blocks = []
        resets = []
        for block_no, (start, end, second) in enumerate(block_spec):
            idx = block_index(start, end)
            x = idx.view("int64").astype(np.float64)
            mwh = np.column_stack([np.interp(x, x_min, mwh_grid[:, j])
                                   for j in range(3)])
            kw = np.column_stack([np.interp(x, x_min, kw_grid[:, j])
                                  for j in range(3)])
            blocks.append((idx, mwh, kw))
            # Nullzeile am Anfang jedes Loggerneustarts - und am Anfang der
            # allerersten Aufzeichnung. Ein Dateiwechsel allein ist kein
            # Neustart und bekommt deshalb keine.
            resets.append(block_no > 0 or file_no == 0)

        n = write_meter_file(out_dir / fname, blocks, resets)
        total_rows += n
        span = f"{blocks[0][0][0]} -> {blocks[-1][0][-1]}"
        size_mb = (out_dir / fname).stat().st_size / 1024 ** 2
        print(f"  {fname:<48} {n:>8} Zeilen  {size_mb:6.1f} MB  {span}")

    n_weather = write_weather_file(out_dir / WEATHER_FILENAME, weather)
    print(f"  {WEATHER_FILENAME:<48} {n_weather:>8} Zeilen")

    print(f"\nFertig: {total_rows} Zaehlerzeilen in {len(FILES)} Dateien.")
    print("Alle Werte sind synthetisch. Naechster Schritt: python -m src.stage1_load")


if __name__ == "__main__":
    main()
