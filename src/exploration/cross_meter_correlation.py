"""Paarweise Korrelation der drei WMZ auf jeder Signalebene.

Berechnet die Pearson- und Spearman-Korrelation zwischen wmz_1/2/3 auf
drei Ebenen:

    1. Roh-kW        (Variante A, ``stage2_hourly.parquet``)
    2. MSTL-Trend    (langsame Komponente, ``stage3_stl.parquet``)
    3. MSTL-Residuum (Variante B, anomalie-relevant, ``stage3_stl.parquet``)

**Warum diese Kennzahl?** Sie beantwortet zwei methodische Fragen, die
fuer die Architektur-Entscheidung *lokal vs. global (Pooling)* zentral sind
(vgl. methodology.md §7.11):

- *Ist ein gemeinsames (gepooltes) Training ueber alle WMZ gerechtfertigt?*
  Pooling profitiert von **niedriger Redundanz** (niedrige Korrelation =
  echtes Mehr-an-Daten) bei zugleich **aehnlicher Struktur**. Hohe
  Korrelation wuerde den Datenmengen-Vorteil zunichtemachen
  (Montero-Manso und Hyndman 2006/2021, global vs. lokal).
- *Ist ein multivariat-gemeinsames Modell bzw. ein korrelationsbasierter
  Detektor (Anomaly Transformer, Xu u. a. 2022) motiviert?* Dieser braucht
  Korrelation **auf der anomalie-relevanten Ebene** (Residuum). Ist sie dort
  ~0, gibt es keine zaehler-uebergreifende Assoziation auszunutzen.

Erwartetes Muster (untersuchtes Gebaeude): hohe Trend-Korrelation (gemeinsamer
Saison-/Temperaturtreiber), aber **nahezu null Residuum-Korrelation** -> die
Abweichungen vom Normalmuster laufen je Zaehler unabhaengig. Die
Korrelations-Kennzahl liefert damit einen empirischen Beleg fuer die
Per-Zaehler-Architektur und grenzt ab, fuer welche Variante (Residuum)
Pooling ueberhaupt sinnvoll ist.

Methodischer Rahmen der Kreuzkorrelation: Box und Jenkins (1976).

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet`` (Roh-kW)
         ``outputs/<dataset>/parquet/stage3_stl.parquet``   (Trend, Residuum)
Ausgabe: ``outputs/<dataset>/figures/cross_meter_correlation.png``
         ``outputs/<dataset>/reports/cross_meter_correlation.csv``
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

# Signalebenen: (Anzeigename, Spalten-Suffix bzw. Quelle). Roh-kW kommt aus
# Stage 2, Trend/Residuum aus Stage 3.
LEVELS = [
    ("Roh-kW (Variante A)", "raw"),
    ("MSTL-Trend", "trend"),
    ("MSTL-Residuum (Variante B)", "residual"),
]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    return p.parse_args()


def level_frame(hourly: pd.DataFrame, stl: pd.DataFrame, level: str) -> pd.DataFrame:
    """3-Spalten-Frame (wmz_1/2/3) fuer eine Signalebene."""
    if level == "raw":
        cols = {w: hourly[f"{w}_kw_mean"] for w in WMZ}
    else:
        cols = {w: stl[f"{w}_{level}"] for w in WMZ}
    return pd.DataFrame(cols)


def correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paarweise Pearson- und Spearman-Matrix (NaN paarweise ausgeschlossen)."""
    return df.corr(method="pearson"), df.corr(method="spearman")


def long_rows(level_name: str, pear: pd.DataFrame, spear: pd.DataFrame) -> list[dict]:
    """Obere Dreiecksmatrix als Langform-Zeilen fuer die CSV."""
    rows = []
    for i, a in enumerate(WMZ):
        for b in WMZ[i + 1:]:
            rows.append({
                "level": level_name,
                "pair": f"{a}__{b}",
                "pearson": round(float(pear.loc[a, b]), 4),
                "spearman": round(float(spear.loc[a, b]), 4),
            })
    return rows


def plot_heatmaps(mats: list[tuple[str, pd.DataFrame]], fig_dir: Path) -> None:
    """Drei Pearson-Heatmaps nebeneinander (eine je Signalebene)."""
    fig, axes = plt.subplots(1, len(mats), figsize=(5 * len(mats), 4.2))
    if len(mats) == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, mats):
        im = ax.imshow(m.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(WMZ)))
        ax.set_yticks(range(len(WMZ)))
        ax.set_xticklabels(WMZ, rotation=45, ha="right")
        ax.set_yticklabels(WMZ)
        ax.set_title(name, fontsize=10, pad=10)
        # Werte annotieren.
        for i in range(len(WMZ)):
            for j in range(len(WMZ)):
                ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Pearson r")
    # y > 1: Suptitle oberhalb der Zeichenflaeche absetzen, sonst ueberlappt
    # die zweite Titelzeile die Subplot-Titel (bbox_inches="tight" erweitert
    # den Bildausschnitt entsprechend).
    fig.suptitle("Paarweise Korrelation der WMZ je Signalebene\n"
                 "(hohe Trend-, ~null Residuum-Korrelation -> Pooling nur auf "
                 "dem Residuum sinnvoll)", fontsize=11, y=1.12)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "cross_meter_correlation.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Paarweise Korrelation der drei Zaehler auf allen drei Signalebenen.

    Gerechnet wird auf Rohsignal, MSTL-Trend und Residuum getrennt. Der
    interessante Befund liegt im Vergleich: Auf Trendebene laufen die Zaehler
    weitgehend parallel (gemeinsames Wetter, gemeinsamer Gebaeudebetrieb), im
    Residuum bricht der Zusammenhang ein. Genau das rechtfertigt die
    Entscheidung, je Zaehler ein eigenes Modell zu trainieren statt eines
    gemeinsamen.
    """
    args = parse_args()
    base = ROOT / "outputs" / args.dataset / "parquet"
    hourly_path = base / "stage2_hourly.parquet"
    stl_path = base / "stage3_stl.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    rep_dir = ROOT / "outputs" / args.dataset / "reports"
    for p, stage in [(hourly_path, "stage2_preprocess"), (stl_path, "stage3_stl")]:
        if not p.is_file():
            raise SystemExit(
                f"Eingang nicht gefunden: {p}\n"
                f"Lauf: python src\\{stage}.py --dataset {args.dataset}")

    hourly = pd.read_parquet(hourly_path)
    stl = pd.read_parquet(stl_path)

    print(f"Dataset: {args.dataset}")
    csv_rows: list[dict] = []
    heatmaps: list[tuple[str, pd.DataFrame]] = []
    for name, level in LEVELS:
        df = level_frame(hourly, stl, level)
        pear, spear = correlations(df)
        heatmaps.append((name, pear))
        csv_rows.extend(long_rows(name, pear, spear))
        print(f"\n=== {name} ===")
        print("Pearson r:")
        print(pear.round(2).to_string())

    out = pd.DataFrame(csv_rows)
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "cross_meter_correlation.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}")
    plot_heatmaps(heatmaps, fig_dir)


if __name__ == "__main__":
    main()
