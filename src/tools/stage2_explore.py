"""Explorations-Plots zur stuendlichen Stage-2-Ausgabe.

Erzeugt vier Uebersichtsabbildungen nach ``outputs/<dataset>/figures/``:

    tagesprofil.png            - mittlere kW je Stunde des Tages und Zaehler
    monatsprofil.png           - mittlere kW je Kalendermonat und Zaehler
    temperatur_korrelation.png - kW gegen Aussentemperatur, mit Pearson-r
    glitch_zeitverlauf.png     - Stunden pro Monat mit mindestens einem Flag

Die Abbildungen sind Plausibilitaetskontrollen, kein Teil der
Produktivpipeline - Stage 3 haengt nicht davon ab.

**Nur ``glitch_zeitverlauf`` steht in der Arbeit** (Abbildung zur
Datenqualitaet in Abschnitt 4.1). Die anderen drei sind Vorlaeufer und
wurden inhaltlich abgeloest von den sorgfaeltigeren Fassungen im
Explorations-Modul: ``exploration/hour_weekday_heatmap.py`` (Tagesgang, dort
zusaetzlich nach Wochentag aufgeloest), ``exploration/monthly_boxplot.py``
(Monatsverlauf, dort mit Streuung statt nur Mittelwert) und
``exploration/temperature_scatter.py`` (Temperatur, dort mit
LOWESS-Glaettung). Sie bleiben als schneller Rundum-Blick nach einem
Stage-2-Lauf erhalten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # src/tools/ -> Projekt-Wurzel
DEFAULT_DATASET = "gebaeude_a"

KW_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]
FLAG_COLS = ["wmz_1_was_flagged", "wmz_2_was_flagged", "wmz_3_was_flagged"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun",
                "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (nur die Datensatz-Wahl)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=("Name des Unterordners unter outputs/. "
                         f"Standard: {DEFAULT_DATASET}"))
    return p.parse_args()


def plot_daily_profile(df: pd.DataFrame, fig_dir: Path) -> None:
    """Tagesgang zeichnen: Mittelwert je Stunde des Tages, ein Panel je Zaehler.

    Mittelt ueber alle vier Jahre; Wochentage und Jahreszeiten verschwinden
    dadurch in der Mittelung. Die nach Wochentag aufgeloeste Fassung steht in
    ``exploration/hour_weekday_heatmap.py``.
    """
    daily = df.groupby(df.index.hour)[KW_COLS].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, KW_COLS):
        ax.plot(daily.index, daily[col], marker="o")
        ax.set_title(col)
        ax.set_xlabel("Stunde des Tages")
        ax.set_ylabel("Mittlerer kW")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.3)
    fig.suptitle("Tagesprofil — 4-Jahres-Mittelwert pro Stunde des Tages")
    fig.tight_layout()
    fig.savefig(fig_dir / "tagesprofil.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_profile(df: pd.DataFrame, fig_dir: Path) -> None:
    """Jahresgang zeichnen: Mittelwert je Kalendermonat, ein Panel je Zaehler.

    Fasst gleiche Monate verschiedener Jahre zusammen und zeigt damit die
    Saisonalitaet, nicht die Drift ueber die Jahre. Wer Strukturbrueche
    sucht, nimmt ``exploration/monthly_boxplot.py`` (Jahr-Monat-Buckets mit
    Streuung).
    """
    monthly = df.groupby(df.index.month)[KW_COLS].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, KW_COLS):
        ax.plot(monthly.index, monthly[col], marker="o")
        ax.set_title(col)
        ax.set_xlabel("Monat")
        ax.set_ylabel("Mittlerer kW")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS, rotation=45)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Monatsprofil — Saisonalität des Heizverbrauchs")
    fig.tight_layout()
    fig.savefig(fig_dir / "monatsprofil.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_correlation(df: pd.DataFrame, fig_dir: Path) -> None:
    """Streudiagramm Heizleistung gegen Aussentemperatur, mit Pearson-r.

    Das ausgewiesene r unterschaetzt die Abhaengigkeit systematisch, weil die
    Beziehung an der Heizgrenztemperatur knickt und damit nicht linear ist.
    Belastbarer sind ``exploration/temperature_scatter.py`` (LOWESS),
    ``exploration/mutual_information.py`` (Transinformation) und
    ``exploration/temperature_regimes.py`` (Regimewechsel).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, KW_COLS):
        sub = df[[col, "temperature"]].dropna()
        ax.scatter(sub["temperature"], sub[col], s=2, alpha=0.08)
        r = sub.corr().iloc[0, 1]
        ax.set_title(f"{col}   (r = {r:+.3f})")
        ax.set_xlabel("Außentemperatur [°C]")
        ax.set_ylabel("kW")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Verbrauch vs. Außentemperatur")
    fig.tight_layout()
    fig.savefig(fig_dir / "temperatur_korrelation.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_glitch_timeline(df: pd.DataFrame, fig_dir: Path) -> None:
    """Zeitverlauf der Datenqualitaet: Stunden mit Fehler-Flag je Monat.

    Diese Abbildung steht in der Arbeit. Sie macht sichtbar, dass die
    Fehler-Flags nicht gleichmaessig verteilt sind, sondern sich in
    Stoerungsphasen buendeln - bei wmz_3 dominiert ein zusammenhaengender
    Block, der die Auswertungsbasis dieses Zaehlers spuerbar verkleinert.
    Gezaehlt wird pro Stunde nur, *ob* ein Flag gesetzt war, nicht wie viele.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in FLAG_COLS:
        has_flag = df[col].astype(int)
        monthly = has_flag.resample("ME").sum()
        label = col.replace("_was_flagged", "")
        ax.plot(monthly.index, monthly.values, marker="o",
                markersize=3, label=label)
    ax.set_title("Datenqualität über die Zeit — "
                 "Stunden pro Monat mit ≥1 Fehler-Flag")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Stunden mit Flag / Monat")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "glitch_zeitverlauf.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Stage-2-Ausgabe laden und alle vier Abbildungen erzeugen."""
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    data_path = out_root / "parquet" / "stage2_hourly.parquet"
    fig_dir = out_root / "figures"

    if not data_path.is_file():
        raise SystemExit(
            f"Stage-2-Ausgabe nicht gefunden: {data_path}\n"
            f"Vorher ausfuehren: python src/stage2_preprocess.py "
            f"--dataset {args.dataset}"
        )

    print(f"Datensatz: {args.dataset}")
    print(f"Lade {data_path}")
    df = pd.read_parquet(data_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_daily_profile(df, fig_dir)
    plot_monthly_profile(df, fig_dir)
    plot_temperature_correlation(df, fig_dir)
    plot_glitch_timeline(df, fig_dir)

    print(f"4 Abbildungen geschrieben nach {fig_dir}")


if __name__ == "__main__":
    main()
