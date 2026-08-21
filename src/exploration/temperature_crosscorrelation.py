"""Kreuzkorrelation Heizleistung <-> Aussentemperatur ueber Lags 0..24 h.

Quantifiziert die **thermische Traegheit** des Gebaeudes: Wie viele Stunden
vergehen, bis sich eine Temperaturaenderung maximal im Heizverbrauch
niederschlaegt? Dazu wird fuer jeden Lag ``l in {0, 1, ..., 24}`` die
Korrelation zwischen der aktuellen Heizleistung und der um ``l`` Stunden
**zurueckliegenden** Aussentemperatur berechnet:

    corr( kW_t , temperature_{t-l} )

Der Lag mit dem betragsmaessig groessten (negativen) Korrelationswert ist
ein empirisches Mass fuer die Verzoegerung zwischen Witterung und
Heizreaktion. Liegt das Maximum bei ``l = 0``, reagiert das Gebaeude
praktisch unmittelbar; ein nach rechts verschobenes Maximum deutet auf
Speichermasse / Traegheit hin.

**Einordnung:** Die Kreuzkorrelationsfunktion ist ein Standardwerkzeug der
Zeitreihenanalyse (Box und Jenkins 1976). Im Gegensatz zum reinen Pearson-r
bei Lag 0 (vgl. ``tools/stage2_explore.py``) macht die lag-aufgeloeste
Betrachtung die Dynamik sichtbar und begruendet, ob lagged-Temperatur-
Features ueberhaupt zusaetzliche Information truegen.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/kreuzkorrelation_temp.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]
MAX_LAG = 24            # bis zu einem Tag Verzoegerung pruefen


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    p.add_argument("--max-lag", type=int, default=MAX_LAG,
                   help=f"Maximaler Lag in Stunden. Default: {MAX_LAG}.")
    return p.parse_args()


def lagged_corr(consumption: pd.Series, temperature: pd.Series,
                max_lag: int) -> pd.Series:
    """Korrelation kW_t vs. temperature_{t-lag} fuer lag = 0..max_lag.

    ``shift(lag)`` verschiebt die Temperatur in die Vergangenheit; die
    Korrelation wird paarweise auf den gemeinsamen Nicht-NaN-Stunden
    gebildet (pandas ``.corr`` ignoriert NaN paarweise).
    """
    out = {}
    for lag in range(max_lag + 1):
        out[lag] = consumption.corr(temperature.shift(lag))
    return pd.Series(out)


def plot_crosscorr(df: pd.DataFrame, fig_dir: Path, max_lag: int) -> None:
    """Korrelation gegen den Temperatur-Lag auftragen, ein Panel je Zaehler.

    Das Maximum der Kurve markiert die Verzoegerung, mit der das Gebaeude auf
    eine Temperaturaenderung reagiert.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in WMZ_COLS:
        wmz = col.replace("_kw_mean", "")
        cc = lagged_corr(df[col], df["temperature"], max_lag)
        best_lag = int(cc.abs().idxmax())
        ax.plot(cc.index, cc.to_numpy(), marker="o", markersize=3,
                label=f"{wmz}  (Extremum bei Lag {best_lag} h, "
                      f"r={cc[best_lag]:+.2f})")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag der Aussentemperatur [h]  (kW_t vs. temp_{t-lag})")
    ax.set_ylabel("Pearson-Korrelation")
    ax.set_title("Kreuzkorrelation Heizleistung <-> Aussentemperatur "
                 "(Box und Jenkins 1976)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "kreuzkorrelation_temp.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Kreuzkorrelation Heizleistung gegen Aussentemperatur ueber Lags 0-24 h.

    Quantifiziert die thermische Traegheit des Gebaeudes: Wie viele Stunden
    vergehen, bis sich ein Temperatursturz maximal im Verbrauch niederschlaegt?
    Das Ergebnis begruendet, warum die Merkmale rollierende Fenster ueber
    mehrere Stunden verwenden statt nur den Momentanwert.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Dataset: {args.dataset}, Stunden: {len(df)}, max-lag: {args.max_lag}")
    plot_crosscorr(df, fig_dir, args.max_lag)


if __name__ == "__main__":
    main()
