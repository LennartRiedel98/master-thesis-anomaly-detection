"""Boxplots der stuendlichen Heizleistung je Kalendermonat (Drift-Diagnose).

Stellt fuer jeden WMZ die Verteilung der stuendlichen kW pro
Jahr-Monat-Bucket als Boxplot-Reihe dar. Damit werden **drei Dinge auf
einen Blick** sichtbar:

* **Saisonale Amplitude** - hohe Mediane/Boxen im Winter, niedrige im
  Sommer (erwartet bei Raumheizung).
* **Langfrist-Drift / Strukturbrueche** - eine ueber die Jahre fallende
  oder springende Box-Lage deutet auf Verbrauchsaenderung hin; relevant
  fuer den im Datensatz dokumentierten EnSikuMaV-Effekt (ab Sep 2022) und
  die COVID-Phase (2020-2021).
* **Ausreisser-Anteil** - die Zahl der Boxplot-Flier (Punkte ausserhalb
  1,5 x IQR) je Monat ist ein roher Indikator fuer anomalie- bzw.
  glitch-behaftete Zaehler.

Der Boxplot wurde von Tukey (1977) in *Exploratory Data Analysis* als
robuste, quartilsbasierte Verteilungsdarstellung eingefuehrt; er ist
gegenueber einzelnen Extremwerten unempfindlicher als Mittelwert +/-
Standardabweichung und damit fuer (potentiell anomalie-behaftete)
Rohdaten geeignet.

Eingang: ``outputs/<dataset>/parquet/stage2_hourly.parquet``
Ausgabe: ``outputs/<dataset>/figures/boxplot_monatlich.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def monthly_groups(series: pd.Series) -> tuple[list[str], list[pd.Series]]:
    """Gruppiert die Reihe nach (Jahr, Monat) und gibt Labels + Werte zurueck.

    Leere Buckets (durch die Aug-Sep-2020-Luecke moeglich) werden
    uebersprungen, damit ``boxplot`` keine leeren Positionen erhaelt.
    """
    s = series.dropna()
    labels: list[str] = []
    data: list[pd.Series] = []
    # ``period`` = monatlicher Zeitstempel; sortiert chronologisch.
    for period, grp in s.groupby(s.index.to_period("M")):
        if len(grp) == 0:
            continue
        labels.append(str(period))      # z. B. '2022-09'
        data.append(grp)
    return labels, data


def plot_boxplots(df: pd.DataFrame, fig_dir: Path) -> None:
    """Drei gestapelte Subplots (ein WMZ je Zeile), gemeinsame x-Zeitachse."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    for ax, col in zip(axes, WMZ_COLS):
        wmz = col.replace("_kw_mean", "")
        labels, data = monthly_groups(df[col])
        ax.boxplot(data, showfliers=True, flierprops={"marker": ".",
                   "markersize": 2, "alpha": 0.3})
        ax.set_xticks(range(1, len(labels) + 1))
        # Nur jeden dritten Monat beschriften, sonst ueberlappt die Achse.
        ax.set_xticklabels([lab if i % 3 == 0 else ""
                            for i, lab in enumerate(labels)],
                           rotation=90, fontsize=7)
        ax.set_ylabel(f"{wmz}  kW")
        ax.grid(True, axis="y", alpha=0.3)
        # EnSikuMaV-Markierung (Sep 2022), falls im Beobachtungsfenster.
        if "2022-09" in labels:
            ax.axvline(labels.index("2022-09") + 1, color="tab:orange",
                       linestyle="--", linewidth=1.0, alpha=0.7,
                       label="EnSikuMaV (Sep 2022)")
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Jahr-Monat")
    fig.suptitle(
        "Monatliche Verteilung der stuendlichen Heizleistung je WMZ "
        "(Boxplot nach Tukey 1977)\nBox = Interquartilsbereich, Linie = "
        "Median, Punkte = Ausreisser (> 1,5 x IQR)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "boxplot_monatlich.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    """Boxplots der Stundenwerte je Jahr-Monat-Bucket, ein Panel je Zaehler.

    Anders als ein Monatsmittel zeigt der Boxplot auch die Streuung. Sichtbar
    werden dadurch drei Dinge zugleich: die saisonale Amplitude, die
    Verschiebung des Niveaus ueber die Jahre (unter anderem der Rueckgang ab
    September 2022) und Monate mit auffaellig zusammengeschrumpfter Streuung -
    ein Hinweis auf Messausfaelle.
    """
    args = parse_args()
    in_path = ROOT / "outputs" / args.dataset / "parquet" / "stage2_hourly.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    if not in_path.is_file():
        raise SystemExit(f"Stage-2-Output nicht gefunden: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Dataset: {args.dataset}, Stunden: {len(df)}")
    plot_boxplots(df, fig_dir)


if __name__ == "__main__":
    main()
