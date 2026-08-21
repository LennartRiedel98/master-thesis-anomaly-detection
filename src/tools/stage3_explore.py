"""Explorations-Plots zur MSTL-Zerlegung aus Stage 3.

Erzeugt fuenf Abbildungsgruppen nach ``outputs/<dataset>/figures/``:

    stl_zoom_<wmz>.png    - Vier-Wochen-Zoom der Zerlegung, je Zaehler
    stl_trend_alle.png    - Trendverlauf ueber vier Jahre, alle drei Zaehler
    stl_tagesprofil.png   - Tagesmuster (Stundenmittel von seasonal_24h)
    stl_wochenprofil.png  - Wochenmuster (Wochentagsmittel von seasonal_168h)
    stl_residual_hist.png - Residuenverteilung je Zaehler (log-skalierte y-Achse)

Bis auf das Residuen-Histogramm stehen alle Abbildungen in der Arbeit
(Kapitel 4.1): der Zoom fuer alle drei Zaehler, der Trendverlauf sowie
Tages- und Wochenprofil.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # src/tools/ -> Projekt-Wurzel
DEFAULT_DATASET = "gebaeude_a"

WMZ = ["wmz_1", "wmz_2", "wmz_3"]
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# Vier-Wochen-Fenster fuer den Zoom. Bewusst mitten in der Heizsaison
# gewaehlt: Im Sommer laeuft wmz_3 nahe null, dort waere von der Zerlegung
# nichts zu sehen.
ZOOM_START = pd.Timestamp("2022-01-10")
ZOOM_END = pd.Timestamp("2022-02-07")


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (nur die Datensatz-Wahl)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    return p.parse_args()


def plot_decomposition_zoom(stage2: pd.DataFrame, stage3: pd.DataFrame,
                            wmz: str, fig_dir: Path) -> None:
    """Die fuenf Ebenen der Zerlegung uebereinander, fuer ein Vier-Wochen-Fenster.

    Von oben nach unten: Originalsignal, Trend, Tagessaison, Wochensaison,
    Residuum. Die gemeinsame Zeitachse macht sichtbar, welcher Anteil des
    Originals von welcher Komponente erklaert wird und was als Residuum
    uebrig bleibt - also das, womit Schiene B arbeitet.
    """
    win = slice(ZOOM_START, ZOOM_END)
    fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(stage2.loc[win, f"{wmz}_kw_mean"], color="black", lw=0.8)
    axes[0].set_ylabel("Original\n[kW]")
    axes[1].plot(stage3.loc[win, f"{wmz}_trend"], color="C0", lw=1.2)
    axes[1].set_ylabel("Trend")
    axes[2].plot(stage3.loc[win, f"{wmz}_seasonal_24h"], color="C1", lw=0.8)
    axes[2].set_ylabel("Saison\n24h")
    axes[3].plot(stage3.loc[win, f"{wmz}_seasonal_168h"], color="C2", lw=0.8)
    axes[3].set_ylabel("Saison\n168h")
    axes[4].plot(stage3.loc[win, f"{wmz}_residual"], color="C3", lw=0.6)
    axes[4].axhline(0, color="grey", lw=0.5)
    axes[4].set_ylabel("Residuum")
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"MSTL-Zerlegung — {wmz}, "
                 f"{ZOOM_START.date()} bis {ZOOM_END.date()}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"stl_zoom_{wmz}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_trend_timeline(stage3: pd.DataFrame, fig_dir: Path) -> None:
    """Trendkomponente aller drei Zaehler ueber den gesamten Zeitraum.

    Die Trendkurve ist die Ebene, auf der Schiene C arbeitet: Hier sind
    Jahresgang, langfristige Niveauverschiebungen und Strukturbrueche
    sichtbar - unter anderem der Rueckgang ab September 2022 (EnSikuMaV) und
    der Ausfallblock bei wmz_3.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, w in zip(axes, WMZ):
        ax.plot(stage3[f"{w}_trend"], color="C0", lw=0.8)
        ax.set_ylabel(f"{w}\n[kW]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Datum")
    fig.suptitle("STL-Trend über den gesamten 4-Jahres-Zeitraum")
    fig.tight_layout()
    fig.savefig(fig_dir / "stl_trend_alle.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_daily_profile(stage3: pd.DataFrame, fig_dir: Path) -> None:
    """Mittleres Tagesmuster aus der 24-h-Saisonkomponente.

    Gemittelt wird die *Saisonkomponente*, nicht das Rohsignal: Trend und
    Niveau sind bereits herausgerechnet, die Kurve schwankt um null und
    zeigt damit die reine Tagesform. Positive Werte bedeuten
    ueberdurchschnittliche, negative unterdurchschnittliche Last zu dieser
    Tagesstunde.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, w in zip(axes, WMZ):
        s = stage3[f"{w}_seasonal_24h"]
        profile = s.groupby(s.index.hour).mean()
        ax.plot(profile.index, profile.values, marker="o")
        ax.set_title(f"{w} — Tagesmuster")
        ax.set_xlabel("Stunde des Tages")
        ax.set_ylabel("Saison-24h-Komponente [kW]")
        ax.set_xticks(range(0, 24, 3))
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Typisches Tagesprofil aus STL-Saisonkomponente")
    fig.tight_layout()
    fig.savefig(fig_dir / "stl_tagesprofil.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_profile(stage3: pd.DataFrame, fig_dir: Path) -> None:
    """Mittleres Wochenmuster aus der 168-h-Saisonkomponente.

    Analog zum Tagesprofil, nur ueber den Wochentag gemittelt. Der
    Wochenend-Abfall ist der Beleg dafuer, dass die 168-h-Periode
    tatsaechlich Nutzungsrhythmus abbildet und nicht nur Rauschen.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, w in zip(axes, WMZ):
        s = stage3[f"{w}_seasonal_168h"]
        profile = s.groupby(s.index.dayofweek).mean()
        ax.plot(profile.index, profile.values, marker="o")
        ax.set_title(f"{w} — Wochenmuster")
        ax.set_xlabel("Wochentag")
        ax.set_ylabel("Saison-168h-Komponente [kW]")
        ax.set_xticks(range(7))
        ax.set_xticklabels(WEEKDAYS)
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Typisches Wochenmuster aus STL-Saisonkomponente")
    fig.tight_layout()
    fig.savefig(fig_dir / "stl_wochenprofil.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution(stage3: pd.DataFrame, fig_dir: Path) -> None:
    """Histogramm der Residuen je Zaehler, y-Achse logarithmisch.

    Die logarithmische Achse ist hier der Punkt: Auf linearer Skala waere
    nur der Zentralbereich zu sehen. Sichtbar werden soll gerade das
    Verhalten der Raender - die schweren Enden sind das Signal, auf das die
    Detektoren der Schiene B reagieren sollen.

    Diese Abbildung dient nur der Diagnose und steht nicht in der Arbeit.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, w in zip(axes, WMZ):
        r = stage3[f"{w}_residual"].dropna()
        ax.hist(r, bins=80, color="C3", alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{w}_residual  "
                     f"(std={r.std():.1f}, n={len(r):,})")
        ax.set_xlabel("Residuum [kW]")
        ax.set_ylabel("Häufigkeit (log)")
        ax.axvline(0, color="grey", lw=0.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Residuum-Verteilung — Input für die AD-Modelle")
    fig.tight_layout()
    fig.savefig(fig_dir / "stl_residual_hist.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Stage-2- und Stage-3-Ausgabe laden und alle sieben Abbildungen erzeugen."""
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    stage2_path = out_root / "parquet" / "stage2_hourly.parquet"
    stage3_path = out_root / "parquet" / "stage3_stl.parquet"
    fig_dir = out_root / "figures"

    for p in (stage2_path, stage3_path):
        if not p.is_file():
            raise SystemExit(f"Benoetigte Eingabe nicht gefunden: {p}")

    print(f"Datensatz: {args.dataset}")
    stage2 = pd.read_parquet(stage2_path)
    stage3 = pd.read_parquet(stage3_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for w in WMZ:
        plot_decomposition_zoom(stage2, stage3, w, fig_dir)
    plot_trend_timeline(stage3, fig_dir)
    plot_daily_profile(stage3, fig_dir)
    plot_weekly_profile(stage3, fig_dir)
    plot_residual_distribution(stage3, fig_dir)

    print(f"7 Abbildungen geschrieben nach {fig_dir}")


if __name__ == "__main__":
    main()
