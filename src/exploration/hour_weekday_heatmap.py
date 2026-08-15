"""Heatmap der mittleren Heizleistung pro (Stunde des Tages x Wochentag).

Stellt fuer jeden WMZ die mittlere stuendliche Heizleistung als
24x7-Heatmap dar (Zeilen = Stunde, Spalten = Wochentag). Diese
sogenannte Wochenprofil-Heatmap (im Energie-Analytics-Kontext auch
``carpet plot``) zeigt **gleichzeitig** das Tages- und Wochenprofil
und macht Asymmetrien (z. B. Wochenend-Abfall, Morgenrampe an Werktagen,
Spaetabend-Spitze) auf einen Blick sichtbar.

Konzeptionell ist das eine Auspraegung der von Tukey (1977) in
*Exploratory Data Analysis* propagierten Idee, mehrdimensionale
Muster ueber Visualisierung statt einzelner Kennzahlen sichtbar zu
machen. In der Gebaeudeenergetik ist dieser Plot Standard fuer
Lastprofil-Diagnose; im vorliegenden Datensatz dient er als
Sanity-Check der MSTL-Periodizitaeten (24 h Tagesprofil + 168 h
Wochenprofil, vgl. Methodik §4.1).

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/heatmap_stunde_wochentag.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "demo_synthetic"

WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]
WEEKDAY_LABELS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (Datensatz-Wahl und Ausgabepfade)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    return p.parse_args()


def build_heatmap(series: pd.Series) -> pd.DataFrame:
    """Mittlere kW pro (hour, weekday). NaN-Stunden werden ignoriert.

    Rueckgabe: 24x7 DataFrame mit Index 0-23 (Stunde) und Spalten 0-6
    (Mo-So). Diese Form erlaubt das direkte Plotten via ``imshow``.
    """
    # Wir bauen ein Hilfs-Frame mit zwei kategorialen Achsen plus
    # Wert; pivot reduziert auf die mittlere kW je Zelle.
    helper = pd.DataFrame({
        "value": series,
        "hour": series.index.hour,
        "weekday": series.index.weekday,   # 0 = Montag, 6 = Sonntag
    }).dropna(subset=["value"])
    return helper.pivot_table(index="hour", columns="weekday",
                              values="value", aggfunc="mean")


def plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    """Die 24x7-Heatmaps zeichnen, ein Panel je Zaehler.

    Zeilen sind Tagesstunden, Spalten Wochentage. Die gemeinsame Farbskala
    innerhalb eines Panels macht Morgenrampe, Abendspitze und
    Wochenendabfall zugleich sichtbar.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, col in zip(axes, WMZ_COLS):
        wmz = col.replace("_kw_mean", "")
        hm = build_heatmap(df[col])
        # ``imshow`` mit origin="upper", damit Stunde 0 oben liegt -
        # passt zur Lese-Konvention (Tageslauf von oben nach unten).
        im = ax.imshow(hm.to_numpy(), aspect="auto", origin="upper",
                       cmap="viridis")
        ax.set_xticks(range(7))
        ax.set_xticklabels(WEEKDAY_LABELS)
        ax.set_yticks(range(0, 24, 3))
        ax.set_yticklabels([f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlabel("Wochentag")
        ax.set_ylabel("Stunde des Tages")
        ax.set_title(f"{wmz}: Wochenprofil [kW]")
        # Farbskala lokal pro Subplot, weil die kW-Niveaus zwischen den
        # WMZ um Faktor ~10 streuen (siehe Stage 2: median(wmz_1)=10,
        # median(wmz_3)=59) - eine gemeinsame Skala wuerde wmz_1 zur
        # einfarbigen Flaeche reduzieren.
        fig.colorbar(im, ax=ax, label="mittlere kW")

    fig.suptitle("Heatmap: mittlere Heizleistung pro (Stunde x Wochentag) "
                 "ueber 4 Jahre (Tukey 1977, EDA)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "heatmap_stunde_wochentag.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Wochenprofil-Heatmap je Zaehler erzeugen (Stunde x Wochentag).

    Zeigt Tages- und Wochenmuster in einem Bild. Das ist die Abbildung, an
    der sich die physikalische Zuordnung der Zaehler ablesen laesst: Der
    Trinkwarmwasser-Zaehler hat ein ausgepraegtes, wochentagsunabhaengiges
    Tagesprofil, die Heizungszaehler ein flacheres mit Wochenendeinbruch.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Dataset: {args.dataset}, Stunden: {len(df)}")
    plot_heatmaps(df, fig_dir)


if __name__ == "__main__":
    main()
