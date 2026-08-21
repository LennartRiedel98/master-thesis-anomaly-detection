"""Trend- und Saisonalitaets-Staerke je WMZ aus der MSTL-Zerlegung.

Berechnet die *strength of trend* und *strength of seasonality* nach der
varianzbasierten Definition von Wang, Smith und Hyndman (2006):

    F_Trend  = max(0, 1 - Var(R) / Var(T + R))
    F_Saison = max(0, 1 - Var(R) / Var(S + R))

mit Trend ``T``, Saisonkomponente ``S`` (hier getrennt fuer die 24-h- und
168-h-Periode) und Residuum ``R`` aus der Stage-3-MSTL-Zerlegung. Die
Masse liegen in ``[0, 1)``: ein Wert nahe 1 bedeutet, dass die jeweilige
Komponente die Reihe dominiert (Residuum klein gegenueber Komponente +
Residuum), ein Wert nahe 0 bedeutet, dass die Komponente kaum Struktur
beitraegt.

**Warum diese Kennzahl?** Sie ersetzt das qualitative Ablesen der
MSTL-Plots durch eine **numerische, zwischen den Zaehlern vergleichbare**
Groesse. Damit laesst sich die in §5.1/§5.3 qualitativ beschriebene
Beobachtung ("wmz_3 hat schwaechere Saisonalitaet / hoeheren Glitch-
Anteil") quantitativ belegen und in einer Tabelle der Thesis berichten.

Eingang: ``outputs/<dataset>/parquet/stage3_stl.parquet``
Ausgabe: ``outputs/<dataset>/figures/stl_strength.png``
         ``outputs/<dataset>/reports/stl_strength.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

WMZ = ["wmz_1", "wmz_2", "wmz_3"]
# Saisonkomponenten, die Stage 3 schreibt (Perioden 24 h und 168 h).
SEASONAL_SUFFIXES = ["seasonal_24h", "seasonal_168h"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    return p.parse_args()


def strength(component: pd.Series, residual: pd.Series) -> float:
    """F = max(0, 1 - Var(R) / Var(component + R)) auf gemeinsamem Index.

    Die Varianzen werden nur auf den Stunden gebildet, an denen sowohl die
    Komponente als auch das Residuum definiert sind (das Residuum ist an
    Daten-Luecken bewusst NaN, siehe Stage 3). ``ddof=0`` (Populations-
    varianz), weil wir die gesamte beobachtete Reihe und keine Stichprobe
    bewerten.
    """
    df = pd.concat([component.rename("c"), residual.rename("r")], axis=1).dropna()
    var_r = df["r"].var(ddof=0)
    var_cr = (df["c"] + df["r"]).var(ddof=0)
    if var_cr <= 0:
        return float("nan")
    return max(0.0, 1.0 - var_r / var_cr)


def compute_table(df: pd.DataFrame) -> pd.DataFrame:
    """Strength-Matrix WMZ x {Trend, Saison_24h, Saison_168h}."""
    rows = []
    for wmz in WMZ:
        resid = df[f"{wmz}_residual"]
        row = {"wmz": wmz,
               "F_trend": strength(df[f"{wmz}_trend"], resid)}
        for suf in SEASONAL_SUFFIXES:
            row[f"F_{suf}"] = strength(df[f"{wmz}_{suf}"], resid)
        rows.append(row)
    return pd.DataFrame(rows).set_index("wmz")


def plot_table(table: pd.DataFrame, fig_dir: Path) -> None:
    """Gruppierte Balken: je WMZ ein Balken-Tripel (Trend/24h/168h)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    measures = ["F_trend", "F_seasonal_24h", "F_seasonal_168h"]
    labels = ["Trend", "Saison 24 h", "Saison 168 h"]
    x = np.arange(len(table.index))
    width = 0.25
    for i, (m, lab) in enumerate(zip(measures, labels)):
        ax.bar(x + (i - 1) * width, table[m].to_numpy(), width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Staerke  F in [0, 1)")
    # pad: Abstand zwischen Titel und Achsenrand, sonst klebt die zweite
    # Titelzeile am obersten Tick-Label (1.0).
    ax.set_title("Trend- und Saisonalitaets-Staerke der MSTL-Komponenten\n"
                 "(Definition nach Wang, Smith und Hyndman 2006)", pad=18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "stl_strength.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Trend- und Saisonalitaets-Staerke je Zaehler aus der MSTL berechnen.

    Verwendet die varianzbasierte Definition nach Wang, Smith und Hyndman:
    F = max(0, 1 - Var(Rest) / Var(Komponente + Rest)). Werte nahe 1 bedeuten,
    dass die jeweilige Komponente das Signal dominiert. Die Kennzahl ist das
    Mass, mit dem die Zaehler-Zuordnung gegengeprueft wurde - der
    Trinkwarmwasser-Zaehler zeigt die staerkste Tagesperiodik.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage3_stl.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    rep_dir = ROOT / "outputs" / args.dataset / "reports"
    if not in_path.is_file():
        raise SystemExit(
            f"Stage-3-Output nicht gefunden: {in_path}\n"
            f"Lauf: python src\\stage3_stl.py --dataset {args.dataset}")
    df = pd.read_parquet(in_path)
    table = compute_table(df)
    print(f"Dataset: {args.dataset}")
    print(table.round(3).to_string())

    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "stl_strength.csv"
    table.round(4).to_csv(csv_path)
    print(f"  Wrote {csv_path}")
    plot_table(table, fig_dir)


if __name__ == "__main__":
    main()
