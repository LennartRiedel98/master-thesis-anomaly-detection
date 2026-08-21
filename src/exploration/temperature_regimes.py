"""Heizgrenzen-Regimewechsel: Niveau UND Streuung je Temperaturklasse.

Der LOWESS-Scatter (``temperature_scatter.py``) zeigt den Knick an der
Heizgrenztemperatur qualitativ. Dieses Skript beziffert ihn und macht einen
zweiten, im Scatter kaum sichtbaren Effekt sichtbar: **die Temperatur steuert
nicht nur das Niveau, sondern auch die Streuung** der Heizleistung.

Genau diese Regimeabhaengigkeit ist der Grund, warum die Rangkorrelation nach
Spearman die Temperaturbindung unterschaetzt, waehrend die Transinformation sie
erfasst - Spearman bewertet nur die Rangordnung der Lage, die Transinformation
die gesamte bedingte Verteilung (vgl. ``mutual_information.py``, MA 4.2.2).

Darstellung: **Small Multiples, ein Panel je WMZ.** Bewusst keine gemeinsame
y-Achse und keine zweite Achse - die drei Zaehler unterscheiden sich um eine
Groessenordnung (Median 9,9 / 4,9 / 59,4 kW), eine geteilte Skala wuerde die
beiden Trinkwarmwasser-Kreise plattdruecken. Jedes Panel traegt Mittelwert
(Linie) und +/- 1 Standardabweichung (Band) je 3-K-Temperaturklasse.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/temperatur_regime.png``
         ``outputs/<dataset>/reports/temperatur_regime.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]
BIN_EDGES = list(range(-12, 33, 3))
HEIZGRENZE = 15.0          # Grad C, Energy-Signature-Knick (Hammarsten 1987)

# Slot 1 der kategorialen Palette; je Panel nur EINE Serie, daher keine
# Legende - der Panel-Titel benennt sie.
LINE_COLOR = "#2a78d6"
BAND_COLOR = "#2a78d6"
MARK_COLOR = "#52514e"


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    return p.parse_args()


def klassen_statistik(df: pd.DataFrame) -> pd.DataFrame:
    """Mittelwert, Std und Besetzung je Temperaturklasse und WMZ."""
    tb = pd.cut(df["temperature"], BIN_EDGES)
    rows = []
    for col in WMZ_COLS:
        g = df.groupby(tb, observed=True)[col]
        for iv, mean in g.mean().items():
            rows.append({
                "wmz": col.replace("_kw_mean", ""),
                "t_von": iv.left, "t_bis": iv.right,
                "t_mitte": (iv.left + iv.right) / 2,
                "mean_kw": mean,
                "std_kw": g.std()[iv],
                "n": int(g.size()[iv]),
            })
    return pd.DataFrame(rows)


def regime_korrelationen(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson-r getrennt unterhalb/oberhalb der Heizgrenze."""
    rows = []
    for col in WMZ_COLS:
        pair = df[[col, "temperature"]].dropna()
        lo = pair[pair["temperature"] < HEIZGRENZE]
        hi = pair[pair["temperature"] >= HEIZGRENZE]
        rows.append({
            "wmz": col.replace("_kw_mean", ""),
            "r_unter_heizgrenze": lo[col].corr(lo["temperature"]),
            "n_unter": len(lo),
            "r_ueber_heizgrenze": hi[col].corr(hi["temperature"]),
            "n_ueber": len(hi),
        })
    return pd.DataFrame(rows).set_index("wmz")


def plot_regimes(stat: pd.DataFrame, korr: pd.DataFrame, fig_dir: Path) -> None:
    """Niveau und Streuung je Temperaturklasse zeichnen.

    Zwei Groessen pro Klasse: der Median als Niveau und die Streuung als
    Breite. Erst die zweite Groesse macht den Regimewechsel sichtbar.
    """
    wmz_list = list(korr.index)
    fig, axes = plt.subplots(1, len(wmz_list), figsize=(13, 4.6))

    for ax, wmz in zip(np.atleast_1d(axes), wmz_list):
        s = stat[stat["wmz"] == wmz].sort_values("t_mitte")
        x = s["t_mitte"].to_numpy()
        m = s["mean_kw"].to_numpy()
        sd = s["std_kw"].to_numpy()

        ax.fill_between(x, np.maximum(m - sd, 0), m + sd,
                        color=BAND_COLOR, alpha=0.18, linewidth=0,
                        label="± 1 Standardabweichung")
        ax.plot(x, m, color=LINE_COLOR, linewidth=2.0, marker="o",
                markersize=4, label="Mittelwert")

        ax.axvline(HEIZGRENZE, color=MARK_COLOR, linestyle="--", linewidth=1.2)
        ax.annotate("Heizgrenze\n15 °C", xy=(HEIZGRENZE, ax.get_ylim()[1]),
                    xytext=(HEIZGRENZE + 1.0, ax.get_ylim()[1] * 0.93),
                    fontsize=8, color=MARK_COLOR, va="top")

        r_lo = korr.loc[wmz, "r_unter_heizgrenze"]
        r_hi = korr.loc[wmz, "r_ueber_heizgrenze"]
        ax.set_title(f"{wmz}\n$r$ = {r_lo:+.2f} (T < 15 °C)   ·   "
                     f"$r$ = {r_hi:+.2f} (T ≥ 15 °C)", fontsize=10)
        ax.set_xlabel("Außentemperatur [°C]")
        ax.grid(color="#d8d8d4", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b4b3ad")

    np.atleast_1d(axes)[0].set_ylabel("Heizleistung [kW]")
    # Unten links: dort verlaeuft in allen Panels keine Kurve (die Last faellt
    # von links oben nach rechts unten), und die Heizgrenzen-Notiz bleibt frei.
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Regimewechsel an der Heizgrenze: Temperatur steuert Niveau "
                 "und Streuung\n(je Panel eigene Skala – die Zähler "
                 "unterscheiden sich um eine Größenordnung)", fontsize=11)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "temperatur_regime.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def main() -> None:
    """Regimewechsel an der Heizgrenze beziffern - Niveau UND Streuung.

    Der LOWESS-Scatter zeigt den Knick an der Heizgrenztemperatur qualitativ;
    hier wird er in Zahlen gefasst. Der zweite, im Scatter kaum sichtbare
    Befund ist der wichtigere: Die Temperatur steuert nicht nur das Niveau der
    Heizleistung, sondern auch ihre Streuung. Genau diese Regimeabhaengigkeit
    ist der Grund, warum ein global kalibrierter Detektor im Uebergangsbereich
    Schwierigkeiten hat.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)

    stat = klassen_statistik(df)
    korr = regime_korrelationen(df)

    print(f"Dataset: {args.dataset}")
    print(korr.round(3).to_string())
    print()
    print("Streuungsverhältnis kalt (< 0 °C) zu warm (> 18 °C):")
    for wmz in korr.index:
        s = stat[stat["wmz"] == wmz].sort_values("t_mitte")["std_kw"].dropna()
        print(f"  {wmz}: {s.iloc[:4].mean() / s.iloc[-4:].mean():5.2f}x")

    rep_dir = ROOT / "outputs" / args.dataset / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv = rep_dir / "temperatur_regime.csv"
    stat.round(3).to_csv(csv, index=False)
    print(f"  Wrote {csv}")

    plot_regimes(stat, korr, ROOT / "outputs" / args.dataset / "figures")


if __name__ == "__main__":
    main()
