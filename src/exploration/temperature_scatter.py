"""Scatter Verbrauch (kW) vs. Aussentemperatur mit LOWESS-Glaettung.

Visualisiert die Beziehung zwischen stuendlichem Wärmeverbrauch und
Aussentemperatur fuer jeden der drei Wärmemengenzähler. Ueberlagert wird
eine LOWESS-Glaettungslinie (locally weighted scatterplot smoothing,
Cleveland 1979): nicht-parametrische, robuste Glaettung, die ohne
Annahme einer bestimmten funktionalen Form lokale Trends abbildet.

**Warum diese Visualisierung?** In der Gebaeudeenergetik ist die
Beziehung Verbrauch ~ Aussentemperatur typischerweise **stueckweise
linear** mit einem Knick am sogenannten *Heizgrenztemperatur-Punkt*
(``balance point``): unterhalb dieser Temperatur steigt der Heizbedarf
naeherungsweise linear mit fallender Temperatur, oberhalb bleibt er
nahe null (Hammarsten 1987 zur Energy-Signature-Methodik). Ein
einfacher Korrelationskoeffizient (wie in ``tools/stage2_explore.py``
gezeigt) verschleiert diesen Knick; ein LOWESS-Smooth bringt ihn
sichtbar heraus, ohne eine bestimmte Knick-Parametrierung anzunehmen.

**LOWESS-Konfiguration:** ``frac=0.3`` (=30 % der Datenpunkte gehen in
jede lokale Regression) ist ein in der Praxis bewaehrter Default fuer
mehrere tausend Punkte; gross genug, um Rauschen zu daempfen, klein
genug, um lokale Strukturen (Heizgrenz-Knick) zu erhalten.
``it=3`` Iterationen (Default in statsmodels) wendet die Tukey-
Bisquare-Robustifizierung an, sodass einzelne Anomaliestunden den Verlauf
nicht verzerren.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/temperatur_scatter_lowess.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# Projekt-Wurzel: src/exploration/temperature_scatter.py -> parents[2].
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

# WMZ-Reihen und ihre menschenlesbaren Titel fuer die Subplots.
WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    p.add_argument("--frac", type=float, default=0.3,
                   help="LOWESS-Glaettungsanteil (Default 0.3). Kleiner = "
                        "lokaler/zerklueftet, groesser = glatter/global.")
    p.add_argument("--exclude-flagged", action="store_true",
                   help="Geflaggte Stunden (wmz_N_was_flagged) von der "
                        "Auswertung ausschliessen (Default: ja, sonst "
                        "Sensor-Glitches verzerren den Smooth).")
    p.set_defaults(exclude_flagged=True)
    return p.parse_args()


def plot_scatter_lowess(df: pd.DataFrame, fig_dir: Path,
                        frac: float, exclude_flagged: bool) -> None:
    """Zeichnet pro WMZ einen Subplot mit Scatter + LOWESS-Linie.

    Wir teilen die Figur zusaetzlich farblich nach Heiz-/Sommer-Saison
    auf (Oktober-April vs. Mai-September): zeigt unmittelbar, dass die
    Heizgrenze nicht nur eine Funktion der Temperatur, sondern auch der
    Jahreszeit ist (Sommer-Punkte liegen tendenziell unterhalb der
    Heizlast-Linie auch bei kuehleren Naechten - die Heizung ist da
    schlicht abgeschaltet).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, col in zip(axes, WMZ_COLS):
        wmz = col.replace("_kw_mean", "")
        # NaN-bereinigte Wertepaare; LOWESS toleriert keine NaN.
        data = df[[col, "temperature"]].dropna()
        if exclude_flagged:
            flag = f"{wmz}_was_flagged"
            if flag in df.columns:
                # Auf den Stunden filtern, die in unserem data-Subset noch
                # vorhanden sind (sonst Index-Mismatch).
                mask = ~df.loc[data.index, flag].fillna(False).astype(bool)
                data = data.loc[mask]

        # Heiz- vs. Sommer-Saison: Oktober-April = Heizen, Mai-September
        # = nicht-Heizen. Wir nutzen den Index-Monat statt einer extra
        # Spalte, weil der DatetimeIndex von Stage 2 das mitliefert.
        is_heating = data.index.month.isin([10, 11, 12, 1, 2, 3, 4])
        ax.scatter(data.loc[is_heating, "temperature"],
                   data.loc[is_heating, col],
                   s=4, alpha=0.15, color="tab:red",
                   label="Heizsaison (Okt-Apr)")
        ax.scatter(data.loc[~is_heating, "temperature"],
                   data.loc[~is_heating, col],
                   s=4, alpha=0.15, color="tab:blue",
                   label="Sommer (Mai-Sep)")

        # LOWESS auf der vollen (saison-vermischten) Reihe, damit die
        # Glaettungslinie die gesamte Heating-Curve darstellt - nicht
        # zwei separate Linien pro Saison. ``return_sorted=True`` (Default)
        # gibt das Ergebnis nach x sortiert zurueck, sodass es direkt
        # plotbar ist.
        sm = lowess(data[col], data["temperature"],
                    frac=frac, it=3, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2.0,
                label=f"LOWESS (frac={frac})")

        ax.set_title(f"{wmz}: Verbrauch vs. Aussentemperatur")
        ax.set_xlabel("Aussentemperatur [°C]")
        ax.set_ylabel("Stunden-Mittel kW")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Heizleistung vs. Aussentemperatur (LOWESS-Glaettung nach Cleveland 1979)\n"
        "Knick markiert die Heizgrenztemperatur (vgl. Hammarsten 1987, "
        "Energy-Signature)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "temperatur_scatter_lowess.png"
    # dpi=140: gut lesbar im Bericht, ohne unnoetig grosse Datei zu
    # produzieren (komplette Figure ca. 250 kB).
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Streudiagramm Heizleistung gegen Aussentemperatur mit LOWESS-Glaettung.

    Die LOWESS-Linie ist nicht-parametrisch, unterstellt also keine
    funktionale Form. Sie macht den Knick an der Heizgrenztemperatur sichtbar:
    Unterhalb steigt die Leistung mit fallender Temperatur, oberhalb laeuft sie
    in ein temperaturunabhaengiges Grundniveau (Trinkwarmwasser). Diese
    Zweiteilung ist die Begruendung dafuer, die Temperatur nicht als einfachen
    linearen Praediktor zu behandeln.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"

    if not in_path.is_file():
        raise SystemExit(
            f"Stage-2-Output nicht gefunden: {in_path}\n"
            f"Lauf: python src\\stage2_preprocess.py --dataset {args.dataset}"
        )
    df = pd.read_parquet(in_path)
    print(f"Dataset: {args.dataset}, Stunden: {len(df)}, "
          f"Temperatur-NaN: {int(df['temperature'].isna().sum())}")
    plot_scatter_lowess(df, fig_dir, args.frac, args.exclude_flagged)


if __name__ == "__main__":
    main()
