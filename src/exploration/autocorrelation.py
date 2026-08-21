"""Autokorrelationsfunktion (ACF) und partielle Autokorrelation (PACF).

Berechnet ACF/PACF auf den drei stuendlichen kW-Reihen aus Stage 2 und
plottet die ersten 7 x 24 = 168 Lags (= eine Woche). Damit ist:

* der **24-h-Peak** als Tages-Saisonalitaet,
* der **168-h-Peak** als Wochen-Saisonalitaet, und
* der grobe Verlauf der Autokorrelation (Persistenz / Decay)

direkt sichtbar. Diese Visualisierung ist die klassische empirische
Begruendung der in Stage 3 (MSTL) angesetzten Saisonalitaeten
``[24, 168]`` und wird in Box und Jenkins (1976) als Standardwerkzeug
der Zeitreihenanalyse eingefuehrt.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/autokorrelation_kw.png``

Implementierungs-Detail: ``statsmodels.graphics.tsaplots`` (``plot_acf``,
``plot_pacf``) braucht eine NaN-freie Eingabe. Wir reichen die Reihe
nach ``.dropna()`` durch - das ist auf der stuendlichen kW-Reihe
unproblematisch, weil die NaN-Quote unter 1 % liegt und die
Autokorrelations-Struktur durch das Dropping nicht systematisch
verzerrt wird.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]
MAX_LAGS = 168          # eine Woche in Stunden
HIGHLIGHT_LAGS = [24, 168]   # Tages- und Wochen-Saisonalitaet


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    p.add_argument("--max-lags", type=int, default=MAX_LAGS,
                   help=f"Anzahl Lags fuer ACF/PACF. Default: {MAX_LAGS} (1 Woche).")
    return p.parse_args()


def plot_acf_pacf(df: pd.DataFrame, fig_dir: Path, max_lags: int) -> None:
    """3 x 2 Grid: pro WMZ je eine ACF- und eine PACF-Spalte.

    ACF zeigt die Korrelation jeder Stunde mit lag-h frueheren Stunden;
    PACF zeigt dieselbe Korrelation nach Herausrechnen aller kuerzeren
    Lags und identifiziert dadurch die direkt wirkenden saisonalen
    Komponenten ohne die durch Persistenz verschobene Korrelation.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for row, col in enumerate(WMZ_COLS):
        s = df[col].dropna()
        plot_acf(s, lags=max_lags, ax=axes[row, 0])
        axes[row, 0].set_title(f"{col.replace('_kw_mean','')} - ACF")
        plot_pacf(s, lags=max_lags, ax=axes[row, 1], method="ywm")
        axes[row, 1].set_title(f"{col.replace('_kw_mean','')} - PACF")
        # Hilfslinien an den Saisonalitaets-Lags, damit visuell klar
        # wird, welche Peaks erwartet werden.
        for lag in HIGHLIGHT_LAGS:
            for ax in (axes[row, 0], axes[row, 1]):
                ax.axvline(lag, color="red", linestyle="--", alpha=0.4,
                           linewidth=0.8)
        # X-Achse: Stunden, gleiche Tick-Punkte fuer Lesbarkeit.
        for ax in (axes[row, 0], axes[row, 1]):
            ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Autokorrelations- und partielle Autokorrelations-Funktion der "
        "stuendlichen kW-Reihen\n(Box und Jenkins 1976; rote Linien: "
        "Lag 24 = Tagesperiode, Lag 168 = Wochenperiode)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "autokorrelation_kw.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """ACF und PACF der drei kW-Reihen ueber eine Woche Lags berechnen.

    Die Abbildung belegt die beiden Perioden, mit denen die MSTL arbeitet:
    Der Ausschlag bei Lag 24 zeigt die Tages-, der bei Lag 168 die
    Wochensaisonalitaet. Sie ist damit die empirische Rechtfertigung fuer die
    Periodenwahl in Stage 3 - diese ist nicht gesetzt, sondern gemessen.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Dataset: {args.dataset}, Stunden: {len(df)}, "
          f"max-lags: {args.max_lags}")
    plot_acf_pacf(df, fig_dir, args.max_lags)


if __name__ == "__main__":
    main()
