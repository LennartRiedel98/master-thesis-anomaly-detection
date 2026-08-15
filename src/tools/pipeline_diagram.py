"""Erzeugt ein Uebersichts-Diagramm der gesamten Pipeline als PNG.

Zeichnet die Stages 1-10 als **horizontalen** Fluss (Landscape, zweizeiliger
"Snake"-Verlauf: Datenaufbereitung oben links->rechts, Modellierung/Bewertung
unten rechts->links), dazu die drei Detektions-Varianten (A/B/C), die fuenf
Detektoren, das Datenexplorations-Modul (Seitenzweig) und die erzeugten
Outputs. Das Bild dient als Thesis-Abbildung und wird nach
``docs/pipeline.png`` geschrieben. Die Datei ist **nicht** im Repo
versioniert - die Abbildung steht in der Arbeit, im Repo steht der Code,
der sie erzeugt.

Aufruf:  python src/tools/pipeline_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # kein Display noetig
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]   # src/tools/ -> Projekt-Wurzel
OUT = ROOT / "docs" / "pipeline.png"

# Farbschema je Bausteinart (pastell, gut druckbar).
COL = {
    "data": "#cfd8dc",      # Datenquellen (grau)
    "stage": "#bbdefb",     # Pipeline-Stages (blau)
    "variant": "#c8e6c9",   # Varianten (gruen)
    "detector": "#ffe0b2",  # Detektoren (orange)
    "evalc": "#e1bee7",     # Evaluation (violett)
    "explore": "#b2dfdb",   # Exploration (tuerkis)
    "out": "#fff9c4",       # Outputs (gelb)
}
EDGE = "#37474f"
ACCENT = "#00695c"
BOX_STYLE = "round,pad=0.02,rounding_size=0.6"


def box(ax, cx, cy, w, h, text, color, fontsize=8.4, bold=False):
    """Abgerundete Box mit weichem Schlagschatten. Gibt die Kanten zurueck."""
    # Schatten (leicht versetzt, halbtransparent).
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2 + 0.35, cy - h / 2 - 0.45), w, h,
        boxstyle=BOX_STYLE, fc="#000000", ec="none", alpha=0.12, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=BOX_STYLE, fc=color, ec=EDGE, lw=1.3, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", zorder=5, linespacing=1.35)
    return {"cx": cx, "cy": cy, "l": cx - w / 2, "r": cx + w / 2,
            "t": cy + h / 2, "b": cy - h / 2}


def connect(ax, p0, p1, color=EDGE, lw=2.0, dashed=False):
    """Pfeil von Punkt p0 nach p1."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                shrinkA=1, shrinkB=1,
                                mutation_scale=18), zorder=4)


def main() -> None:
    """Das Uebersichtsdiagramm der Pipeline zeichnen und nach docs/ schreiben.

    Stellt die Stages 1 bis 10 als horizontalen Fluss dar (zweizeilig, damit
    es im Querformat lesbar bleibt), dazu die drei Schienen, die sechs
    Detektoren, das Explorations-Modul als Seitenzweig und die erzeugten
    Ausgaben.
    """
    fig, ax = plt.subplots(figsize=(19, 10))
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 72)
    ax.axis("off")

    W, H = 12.0, 9.0
    XS = [7, 20.5, 34, 47.5, 61, 74.5, 88]   # 7 Spalten
    Y_TOP, Y_BOT = 47.0, 16.0                 # zwei Zeilen

    # --- Zeile 1 (oben, links -> rechts): Stages 1-6 ---
    row_a = [
        ("Datenquellen\n3× WMZ-CSV (kW, MWh)\n· Open-Meteo (Wetter)", "data"),
        ("Stage 1\nLaden & Mergen\ngemeinsamer Zeitindex", "stage"),
        ("Stage 2\nVorverarbeitung\nFehlerfilter → kW/h, Flags", "stage"),
        ("Stage 3\nMSTL-Zerlegung\nTrend · Saison · Residuum", "stage"),
        ("Stage 4\nFeature Engineering\nZeit (sin/cos), Wetter", "stage"),
        ("Stage 5\nSplit (zeitlich)\nTrain / Val / Test", "stage"),
        ("Stage 6\nNormalisierung\nStandardScaler je Variante", "stage"),
    ]
    a = [box(ax, XS[i], Y_TOP, W, H, t, COL[c]) for i, (t, c) in enumerate(row_a)]

    # --- Zeile 2 (unten, rechts -> links): Varianten, Training, Detektoren,
    #     Injektion, Evaluation, Outputs ---
    row_b = [
        ("Drei Varianten\nA Roh · B Residuum · C Trend", "variant", True),
        ("Stage 7 + Stage 8\nTraining & HPO\nGrid · Random · gridwh", "stage", False),
        ("Sechs Detektoren\nZ-Score · LOF · IForest · Constancy\nPELT · LSTM-AE", "detector", True),
        ("Stage 9\nAnomalie-Injektion\nTest-Set (6 Typen)", "stage", False),
        ("Stage 10\nEvaluation\nP/R/F1 · ROC-/PR-AUC", "evalc", False),
        ("Outputs\nmetrics.csv · figures\nbest_hparams · Bericht", "out", True),
    ]
    xs_b = [88, 74.5, 61, 47.5, 34, 20.5]    # gleiche Spalten, rueckwaerts
    b = [box(ax, xs_b[i], Y_BOT, W, H, t, COL[c], bold=bd)
         for i, (t, c, bd) in enumerate(row_b)]

    # --- Pfeile ---
    for i in range(len(a) - 1):                         # Zeile 1: -> rechts
        connect(ax, (a[i]["r"], Y_TOP), (a[i + 1]["l"], Y_TOP))
    connect(ax, (a[-1]["cx"], a[-1]["b"]), (b[0]["cx"], b[0]["t"]))  # U-Turn
    for i in range(len(b) - 1):                         # Zeile 2: -> links
        connect(ax, (b[i]["l"], Y_BOT), (b[i + 1]["r"], Y_BOT))

    # --- Seitenzweig: Datenexploration (liest Stage 2/3) ---
    ex = box(ax, 47.5, 63.0, 62, 8.2,
             "Datenexploration (§ 5.4) — neun Skripte\n"
             "LOWESS-Scatter · Wochenprofil-Heatmap · ACF/PACF · Jahresdauerlinie · "
             "Monats-Boxplot\nKreuzkorrelation · STL-Strength · Mutual Information · "
             "Cross-Meter-Korrelation",
             COL["explore"], fontsize=8.2)
    connect(ax, (a[3]["cx"], a[3]["t"]), (a[3]["cx"], ex["b"]),
            color=ACCENT, dashed=True, lw=1.6)

    # --- Phasen-Beschriftung ---
    ax.text(1.0, 54.2, "① Datenaufbereitung — Stages 1–6",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#1565c0")
    ax.text(1.0, 9.2, "② Modellierung, Detektion & Bewertung — Stages 7–10",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#6a1b9a")

    # --- Titel ---
    ax.text(48, 70.2,
            "Anomaliedetektion in WMZ-Heizenergiedaten — Pipeline (Stages 1–10)",
            ha="center", va="top", fontsize=15.5, fontweight="bold")

    # --- Legende (horizontale Farbtafel unten) ---
    legend = [("Daten", "data"), ("Pipeline-Stage", "stage"),
              ("Variante", "variant"), ("Detektor", "detector"),
              ("Evaluation", "evalc"), ("Exploration", "explore"),
              ("Output", "out")]
    lx = 7.0
    for label, ckey in legend:
        ax.add_patch(FancyBboxPatch((lx, 3.0), 1.8, 1.8,
                     boxstyle="round,pad=0.02,rounding_size=0.3",
                     fc=COL[ckey], ec=EDGE, lw=1.0, zorder=3))
        ax.text(lx + 2.3, 3.9, label, ha="left", va="center", fontsize=8.6)
        lx += 12.4

    # --- Fussnote ---
    ax.text(95.5, 0.6,
            "A/B: stationäre Anomalien · C: nicht-stationäre (disjunkt, § 6.3)   |   "
            "Constancy (nur A): schließt die Plateau-Lücke   |   "
            "Klassik: CPU · LSTM-AE: GPU   |   outputs/ reproduzierbar (gitignored)",
            ha="right", va="bottom", fontsize=8.2, color="#455a64", style="italic")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
