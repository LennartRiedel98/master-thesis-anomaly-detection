"""Jahresdauerlinie (Load Duration Curve) der drei Heizleistungs-Reihen.

Sortiert die stuendliche Heizleistung jedes WMZ **absteigend** und traegt
sie gegen den kumulierten Anteil der Stunden auf (0 % = Spitzenlast-Stunde,
100 % = schwaechste Stunde). Diese Jahresdauerlinie ist im Energie-
Engineering der Standard, um **Spitzenlast-Konzentration vs. Grundlast**
abzulesen: ein steiler linker Rand mit flachem Plateau bedeutet, dass die
Spitzenlast nur in wenigen Stunden auftritt (typisch fuer wetterabhaengige
Heizlast), waehrend eine flache Kurve eine konstante Grundlast anzeigt
(typisch fuer Warmwasser-/Prozesslast).

**Warum diese Visualisierung?** Sie ordnet die drei Zaehler ohne Annahme
einer funktionalen Form in das Schema Grundlast/Spitzenlast ein und
stuetzt damit die Interpretation aus §5.1/§5.3 (wmz_3 stark
temperaturgetrieben = Raumheizung mit ausgepraegter Spitze; wmz_1/2 mit
hoeherem Grundlast-Sockel). Die Flaeche unter der Kurve entspricht dem
Jahres-Waermebedarf. Die Methode ist fuer Waermenetze in Verbruggen (1980)
als Standard-Lastdauerlinie eingefuehrt; das aktuelle Fernwaerme-Lehrbuch
Frederiksen und Werner (2013) ordnet Lastvariationen und Dauerlinien
ein.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/jahresdauerlinie.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "demo_synthetic"

WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    return p.parse_args()


def duration_curve(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Absteigend sortierte Werte + zugehoeriger Stunden-Anteil [0, 1].

    NaN-Stunden werden verworfen (sie haben keine definierte Last); der
    Stunden-Anteil bezieht sich daher auf die Zahl gueltiger Messstunden.
    """
    vals = np.sort(series.dropna().to_numpy())[::-1]
    frac = np.arange(1, len(vals) + 1) / len(vals)
    return frac, vals


def plot_duration_curves(df: pd.DataFrame, fig_dir: Path) -> None:
    """Eine Figur mit drei Subplots (lokale y-Skala je WMZ).

    Lokale Skala, weil die kW-Niveaus zwischen den Zaehlern um Faktor ~10
    streuen (median wmz_1 ~10, wmz_3 ~59) - eine gemeinsame Achse wuerde
    die Form der kleineren Reihen unkenntlich machen.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, WMZ_COLS):
        wmz = col.replace("_kw_mean", "")
        frac, vals = duration_curve(df[col])
        ax.plot(frac * 100.0, vals, color="tab:red", linewidth=1.6)
        ax.fill_between(frac * 100.0, vals, alpha=0.15, color="tab:red")
        # Grundlast-Indikator: Last, die in >= 95 % der Stunden erreicht
        # wird (Wert bei Stunden-Anteil 0.95).
        base = float(np.quantile(vals, 0.05))   # 5 %-Quantil = 95 %-Dauer
        ax.axhline(base, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=f"Grundlast (95 %-Dauer) = {base:.1f} kW")
        ax.set_title(f"{wmz}: Jahresdauerlinie")
        ax.set_xlabel("Anteil der Stunden [%]")
        ax.set_ylabel("Heizleistung [kW]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Jahresdauerlinien der Heizleistung (sortiert absteigend; "
        "Verbruggen 1980)\nsteiler Abfall = wetterabhaengige Spitzenlast, "
        "flaches Plateau = konstante Grundlast",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "jahresdauerlinie.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Jahresdauerlinie je Zaehler zeichnen (Last absteigend sortiert).

    Die Stundenwerte werden absteigend sortiert und gegen den kumulierten
    Stundenanteil aufgetragen. Ein steiler linker Rand mit flachem Plateau
    bedeutet, dass die Spitzenlast nur in wenigen Stunden auftritt - typisch
    fuer wetterabhaengige Heizung. Ein flacher Verlauf steht fuer Grundlast.
    Die Form unterscheidet die Zaehler deutlich und stuetzt ihre
    Funktionszuordnung.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Dataset: {args.dataset}, Stunden: {len(df)}")
    plot_duration_curves(df, fig_dir)


if __name__ == "__main__":
    main()
