"""Wirkung von Makro-Ereignissen auf den Waermeverbrauch sichtbar machen.

Zwei Abbildungen nach ``outputs/<dataset>/figures/``:

    monatlich_makro_events.png  - Monatsmittel aller drei Zaehler mit
                                  senkrechten Markierungen an den
                                  COVID-Lockdowns und an der EnSikuMaV
                                  (Energiesparverordnung ab September 2022);
                                  der STL-Trend wird ueberlagert, sofern die
                                  Stage-3-Ausgabe vorliegt.
    heizsaison_vergleich.png    - Heizsaisons (Okt-Apr) uebereinandergelegt,
                                  je Zaehler ein Panel, sodass der Rueckgang
                                  nach der EnSikuMaV direkt ablesbar wird.

Dazu eine Konsolentabelle mit Mittelwert und Summe je Heizsaison und Zaehler.

Hintergrund: Der Vier-Jahres-Zeitraum (2019-11 bis 2023-11) ueberschneidet
sich mit den COVID-Lockdowns und der Energiekrise 2022. Beides stoert jede
Definition von "Normalbetrieb" - die Abbildungen beziffern, wie stark.

Die beiden Abbildungen stehen **nicht** in der Arbeit; die
EnSikuMaV-Argumentation stuetzt sich dort auf die Change-Points von PELT
(Schiene C). Gebraucht werden sie fuer den Abgleich der
Falsch-Positiv-Kandidaten in der qualitativen Analyse: Schlaegt ein Detektor
ausserhalb jeder injizierten Anomalie an, ist hier zu pruefen, ob an dieser
Stelle ein reales Makro-Ereignis liegt (siehe ``qualitative_evaluierung.md``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # src/tools/ -> Projekt-Wurzel
DEFAULT_DATASET = "gebaeude_a"

WMZ_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]

MACRO_EVENTS = [
    ("2020-03-22", "1. Lockdown"),
    ("2020-11-02", "Lockdown light"),
    ("2021-12-15", "Omikron-Welle"),
    ("2022-02-24", "Beginn Ukrainekrieg"),
    ("2022-09-01", "EnSikuMaV — max 19 °C"),
    ("2023-04-15", "Auslaufen EnSikuMaV (tlw.)"),
]

HEATING_MONTH_ORDER = [10, 11, 12, 1, 2, 3, 4]
HEATING_MONTH_LABELS = ["Okt", "Nov", "Dez", "Jan", "Feb", "Mär", "Apr"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (nur die Datensatz-Wahl)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    return p.parse_args()


def heating_season_label(ts: pd.Timestamp) -> str:
    """Heizsaison eines Zeitstempels als 'JJJJ/JJ' zurueckgeben.

    Eine Saison laeuft von Oktober bis September: Der Dezember 2022 und der
    Februar 2023 gehoeren beide zu "2022/23". Ohne diese Verschiebung wuerde
    jede Heizperiode von der Jahresgrenze zerschnitten und liesse sich nicht
    als Einheit vergleichen.
    """
    if ts.month >= 10:
        return f"{ts.year}/{(ts.year + 1) % 100:02d}"
    return f"{ts.year - 1}/{ts.year % 100:02d}"


def plot_monthly_timeseries(stage2: pd.DataFrame,
                            stage3: pd.DataFrame | None,
                            fig_dir: Path) -> None:
    """Monatsmittel je Zaehler mit senkrechten Markern an den Makro-Ereignissen.

    Der STL-Trend wird ueberlagert, sofern ``stage3`` vorliegt - er ist
    glatter als das Monatsmittel und trennt die langfristige Verschiebung
    besser vom Jahresgang. Es werden nur Ereignisse eingezeichnet, die in den
    Datenzeitraum fallen.
    """
    monthly = stage2[WMZ_COLS].resample("ME").mean()
    trend_monthly = None
    if stage3 is not None:
        trend_cols = [c.replace("_kw_mean", "_trend") for c in WMZ_COLS]
        trend_monthly = stage3[trend_cols].resample("ME").mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, col in zip(axes, WMZ_COLS):
        ax.plot(monthly.index, monthly[col], marker="o", markersize=3,
                color="C0", label="Monatsmittel")
        if trend_monthly is not None:
            t_col = col.replace("_kw_mean", "_trend")
            ax.plot(trend_monthly.index, trend_monthly[t_col],
                    color="C3", lw=1.2, label="STL-Trend (Monatsmittel)")

        ymax = monthly[col].max() * 1.05
        for date, label in MACRO_EVENTS:
            ts = pd.Timestamp(date)
            if monthly.index[0] <= ts <= monthly.index[-1]:
                ax.axvline(ts, color="grey", lw=0.6, linestyle="--", alpha=0.7)
                ax.text(ts, ymax, label, rotation=90, va="top",
                        ha="right", fontsize=7, color="grey")

        ax.set_ylabel(f"{col}\n[mean kW]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Datum")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Monatlicher Mittelwert mit Makro-Ereignissen "
                 "(Lockdowns, Energiekrise, EnSikuMaV)")
    fig.tight_layout()
    fig.savefig(fig_dir / "monatlich_makro_events.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_heating_season_overlay(stage2: pd.DataFrame, fig_dir: Path) -> None:
    """Heizsaisons uebereinanderlegen, damit sie direkt vergleichbar werden.

    Jede Saison wird als eigene Kurve ueber die Monatsfolge Okt-Apr gelegt.
    Erst dadurch laesst sich der EnSikuMaV-Effekt vom normalen Jahresgang
    trennen: Ein kalter Januar sieht in der Zeitreihe wie ein Anstieg aus,
    hier faellt er mit den Januaren der anderen Saisons zusammen.

    Die Saison 2022/23 ist dicker gezeichnet (die erste unter der
    Verordnung). Saisons mit weniger als sechs belegten Monaten werden als
    "partiell" gekennzeichnet - der Datensatz beginnt im November und endet
    im November, Anfang und Ende sind also unvollstaendig.
    """
    df = stage2[WMZ_COLS].copy()
    df["season"] = [heating_season_label(ts) for ts in df.index]
    df["month"] = df.index.month

    monthly_means = (df.groupby(["season", "month"])[WMZ_COLS]
                       .mean()
                       .reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    seasons = sorted(monthly_means["season"].unique())
    for ax, col in zip(axes, WMZ_COLS):
        for season in seasons:
            sub = monthly_means[monthly_means["season"] == season]
            # Auf die Reihenfolge Okt-Apr umindizieren
            indexed = sub.set_index("month")[col].reindex(HEATING_MONTH_ORDER)
            n_valid = indexed.notna().sum()
            label = season if n_valid >= 6 else f"{season} (partiell)"
            lw = 2.0 if season == "2022/23" else 1.0
            ax.plot(range(7), indexed.values, marker="o", label=label, lw=lw)
        ax.set_title(col)
        ax.set_xticks(range(7))
        ax.set_xticklabels(HEATING_MONTH_LABELS)
        ax.set_ylabel("Mittelwert [kW]")
        ax.set_xlabel("Monat der Heizsaison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Heizsaisons im Vergleich (Okt-Apr) - "
                 "EnSikuMaV-Saison 2022/23 hervorgehoben")
    fig.tight_layout()
    fig.savefig(fig_dir / "heizsaison_vergleich.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_heating_season_summary(stage2: pd.DataFrame) -> None:
    """Kennzahlen je Heizsaison auf die Konsole schreiben.

    Ausgegeben werden Mittelwert und Summe je Zaehler, dazu die Zahl der
    belegten Stunden - Letztere ist noetig, um die unvollstaendigen
    Rand-Saisons nicht mit den vollen zu verwechseln. Die Summen sind in MWh
    umgerechnet (Stundenwerte in kW aufsummiert und durch 1000 geteilt).
    """
    df = stage2[WMZ_COLS].copy()
    df["season"] = [heating_season_label(ts) for ts in df.index]
    df["month"] = df.index.month
    heating = df[df["month"].isin(HEATING_MONTH_ORDER)]

    print("\nHeizsaisons-Vergleich (nur Monate Okt-Apr berücksichtigt):")
    print()
    summary_mean = heating.groupby("season")[WMZ_COLS].mean().round(1)
    summary_total = (heating.groupby("season")[WMZ_COLS].sum() / 1000).round(1)
    counts = heating.groupby("season").size()

    print(f"  {'Saison':<10} {'Stunden':>9}   "
          + "   ".join(f"{c.replace('_kw_mean', ''):>9}" for c in WMZ_COLS))
    print(f"  {'':<10} {'':>9}   "
          + "   ".join(f"{'[mean kW]':>9}" for _ in WMZ_COLS))
    print()
    print("  Mittelwerte:")
    for season in summary_mean.index:
        cols = "   ".join(f"{summary_mean.loc[season, c]:>9.1f}" for c in WMZ_COLS)
        print(f"  {season:<10} {counts[season]:>9d}   {cols}")
    print()
    print("  Heizsaison-Summen (MWh):")
    for season in summary_total.index:
        cols = "   ".join(f"{summary_total.loc[season, c]:>9.1f}" for c in WMZ_COLS)
        print(f"  {season:<10} {'':<9}   {cols}")


def main() -> None:
    """Beide Abbildungen erzeugen und die Saison-Tabelle ausgeben.

    Die Stage-3-Ausgabe ist optional: Fehlt sie, entfaellt lediglich die
    ueberlagerte Trendkurve, alles andere laeuft unveraendert durch.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    stage2_path = out_root / "parquet" / "stage2_hourly.parquet"
    stage3_path = out_root / "parquet" / "stage3_stl.parquet"
    fig_dir = out_root / "figures"

    if not stage2_path.is_file():
        raise SystemExit(f"Stage-2-Ausgabe nicht gefunden: {stage2_path}")

    print(f"Datensatz: {args.dataset}")
    stage2 = pd.read_parquet(stage2_path)
    stage3 = pd.read_parquet(stage3_path) if stage3_path.is_file() else None
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_monthly_timeseries(stage2, stage3, fig_dir)
    plot_heating_season_overlay(stage2, fig_dir)
    print_heating_season_summary(stage2)

    print(f"\n2 Abbildungen geschrieben nach {fig_dir}")


if __name__ == "__main__":
    main()
