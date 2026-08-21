"""Mutual Information Heizleistung <-> Aussentemperatur (vs. Pearson-r).

Der Pearson-Korrelationskoeffizient misst nur den **linearen** Zusammenhang
und unterschaetzt die Temperaturabhaengigkeit systematisch, weil die
Heizleistung-Temperatur-Beziehung wegen der Heizgrenztemperatur
**stueckweise linear / nicht-linear** ist (vgl. ``temperature_scatter.py``).
Die Transinformation (Mutual Information, MI) erfasst dagegen **beliebige**
statistische Abhaengigkeiten und ist daher das ehrlichere Mass fuer die
Frage, wie viel die Aussentemperatur ueber den Verbrauch verraet.

Dieses Skript stellt je WMZ MI und |Pearson-r| gegenueber. Eine Konstellation
"hohe MI bei niedrigem |r|" ist der direkte empirische Beleg, dass Pearson-r
die (nicht-lineare) Temperaturabhaengigkeit unterschaetzt - und damit die
Rechtfertigung, Temperatur als Feature fuer Modelle bereitzustellen, die
Nicht-Linearitaeten lernen koennen (IForest, LSTM-AE), nicht nur fuer
lineare/Score-Verfahren.

**Schaetzer:** ``sklearn.feature_selection.mutual_info_regression`` nutzt den
nichtparametrischen k-Naechste-Nachbarn-Schaetzer nach Kraskov, Stoegbauer
und Grassberger (2004); MI wird in *nats* (natuerlicher Logarithmus)
zurueckgegeben. Ein fixer ``random_state`` macht die (durch das kNN-Sampling
leicht stochastische) Schaetzung reproduzierbar.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/mutual_information_temp.png``
         ``outputs/<dataset>/reports/mutual_information_temp.csv``
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
MI_SEED = 0          # Reproduzierbarkeit des kNN-MI-Schaetzers


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    return p.parse_args()


def mi_and_r(consumption: pd.Series,
             temperature: pd.Series) -> tuple[float, float, float]:
    """MI (nats), Pearson-r und Spearman-rho auf gemeinsamem Nicht-NaN-Index.

    Lazy-Import von sklearn (konsistent mit den Modell-Modulen), damit das
    Skript ohne sklearn importierbar bleibt, falls nur andere Explorationen
    laufen.
    """
    from sklearn.feature_selection import mutual_info_regression

    pair = pd.concat([consumption.rename("y"), temperature.rename("t")],
                     axis=1).dropna()
    x = pair[["t"]].to_numpy()
    y = pair["y"].to_numpy()
    mi = float(mutual_info_regression(x, y, random_state=MI_SEED)[0])
    r = float(pair["y"].corr(pair["t"]))
    rho = float(pair["y"].corr(pair["t"], method="spearman"))
    return mi, r, rho


def r_equivalent(mi_nats: float) -> float:
    """Gausssches Korrelationsaequivalent der Transinformation.

    Fuer bivariat normalverteilte Groessen gilt MI = -0.5*ln(1 - r^2), also
    umgekehrt r = sqrt(1 - exp(-2*MI)). Das bringt die unnormierte nats-Skala
    auf dieselbe [0,1]-Skala wie die Korrelationskoeffizienten und macht den
    Ein-Achsen-Vergleich der drei Masse erst zulaessig.
    """
    return float(np.sqrt(1.0 - np.exp(-2.0 * mi_nats)))


# Kategoriale Farben in fester Reihenfolge (nicht zyklisch), auf CVD-Trennung
# geprueft: paarweise OKLab-dE >= 9,2 unter Protanopie/Deuteranopie,
# Normalsicht-dE >= 24. Reihenfolge = Reihenfolge der Annahmen-Staffelung.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_compare(table: pd.DataFrame, fig_dir: Path) -> None:
    """Drei Masse je WMZ auf EINER Achse: |Pearson-r|, |Spearman-rho|, r_aequiv.

    Bewusst *keine* zweite y-Achse: MI in nats und |r| in [0,1] waeren nicht
    vergleichbar, und zwei Balken nebeneinander laden trotz Disclaimer genau
    zu diesem Vergleich ein - die relative Balkenhoehe waere dann ein Artefakt
    der Achsenwahl. Stattdessen wird die MI ueber das gaussche
    Korrelationsaequivalent r_aequiv = sqrt(1 - exp(-2*MI)) auf dieselbe
    [0,1]-Skala gebracht. Damit ist der Vergleich legitim und die eigentliche
    Aussage - die Staffelung |r| <= |rho| <= r_aequiv - direkt ablesbar.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(table.index))
    width = 0.26
    gap = 0.015  # Flaechen-Abstand zwischen benachbarten Balken

    serien = (
        ("abs_pearson_r", "|Pearson-$r$|  – setzt Linearität voraus"),
        ("abs_spearman_rho", "|Spearman-$\\rho$|  – setzt nur Monotonie voraus"),
        ("r_equiv", "$r_\\mathrm{äquiv}$ aus MI  – setzt nichts voraus"),
    )
    for k, (col, label) in enumerate(serien):
        pos = x + (k - 1) * (width + gap)
        vals = table[col].to_numpy()
        ax.bar(pos, vals, width, color=SERIES_COLORS[k], label=label,
               edgecolor="white", linewidth=0.8)
        # Direkte Wertelabels: Identitaet nie allein ueber Farbe, und die
        # Zahlen sind hier die eigentliche Aussage.
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#52514e")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Zusammenhangsstärke (einheitliche Skala)")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_title(
        "Temperaturabhängigkeit je Wärmemengenzähler: drei Maße mit\n"
        "gestaffelt schwächeren Annahmen (MI-Schätzer: Kraskov u. a. 2004)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", color="#d8d8d4", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b4b3ad")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "mutual_information_temp.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Transinformation zwischen Heizleistung und Aussentemperatur berechnen.

    Stellt sie dem Betrag des Pearson-Koeffizienten gegenueber. Der Vergleich
    ist der eigentliche Punkt: Pearson misst nur den linearen Anteil und
    unterschaetzt die Temperaturabhaengigkeit systematisch, weil die Beziehung
    an der Heizgrenztemperatur knickt. Die Transinformation erfasst beliebige
    Abhaengigkeiten und faellt entsprechend hoeher aus.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    rep_dir = ROOT / "outputs" / args.dataset / "reports"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)

    rows = []
    for col in WMZ_COLS:
        mi, r, rho = mi_and_r(df[col], df["temperature"])
        rows.append({"wmz": col.replace("_kw_mean", ""),
                     "mutual_info_nats": mi, "pearson_r": r,
                     "abs_pearson_r": abs(r), "spearman_rho": rho,
                     "abs_spearman_rho": abs(rho),
                     "r_equiv": r_equivalent(mi)})
    table = pd.DataFrame(rows).set_index("wmz")
    print(f"Dataset: {args.dataset}")
    print(table.round(3).to_string())

    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "mutual_information_temp.csv"
    table.round(4).to_csv(csv_path)
    print(f"  Wrote {csv_path}")
    plot_compare(table, fig_dir)


if __name__ == "__main__":
    main()
