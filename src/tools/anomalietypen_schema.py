"""Übersichts-Schema Anomalietypen für Kap. 2.1.3 der Thesis.

Konzeptdiagramm (keine Daten): Grundtypen nach Chandola u. a. 2009
(Punkt / kontextuell / kollektiv) x Stationaritätsachse der Arbeit
(stationär / nicht-stationär), mit den konkreten WMZ-Fehlerbildern in
den Zellen und der Zuordnung zu den Detektions-Schienen A/B (stationär)
bzw. C (nicht-stationär).

Farben: helles Blau (stationär) / helles Orange (nicht-stationär) -
CVD-taugliches Paar; die Zuordnung steht zusätzlich als Text in den
Spaltenköpfen (Farbe ist nie alleiniger Informationsträger).

Ausgabe: outputs/<dataset>/figures/anomalietypen_schema.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

C_STAT = "#c6dbef"      # helles Blau  (stationär -> Schienen A/B)
C_NONSTAT = "#fdd0a2"   # helles Orange (nicht-stationär -> Schiene C)
C_EDGE = "#666666"
C_HEAD = "#f0f0f0"


def box(ax, x, y, w, h, text, fc, fontsize=10, weight="normal"):
    """Ein abgerundetes Rechteck mit zentriertem Text ins Achsensystem setzen.

    Hilfsfunktion des Schaubilds: Alle Kaesten sollen gleich aussehen, das
    Schema besteht aus rund einem Dutzend davon.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=fc, edgecolor=C_EDGE, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)


def main() -> None:
    """Das Uebersichtsschema der Anomalietypen zeichnen und speichern.

    Reines Konzeptdiagramm ohne Daten: die Grundtypen nach Chandola
    (Punkt, kontextuell, kollektiv) gekreuzt mit der Stationaritaetsachse
    dieser Arbeit, in den Zellen die konkreten Fehlerbilder und die Zuordnung
    zu den Schienen A/B (stationaer) und C (nicht-stationaer).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    args = p.parse_args()
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Raster: linke Labelspalte + 2 Datenspalten, Kopfzeile + 3 Zeilen
    x0, xw = 0.02, 0.20            # Labelspalte
    c1, c2 = 0.24, 0.62            # Spaltenstarts
    cw = 0.36                      # Spaltenbreite
    rh = 0.185                     # Zeilenhöhe
    rows_y = [0.50, 0.29, 0.08]    # Punkt / kontextuell / kollektiv
    head_y = 0.72

    # Kopfzeile (Achse 2: Stationarität + Schienen-Zuordnung)
    box(ax, c1, head_y, cw, 0.16,
        "stationär\n(zeitlich begrenzte Auslenkung)\n→ Schienen A und B",
        C_STAT, fontsize=10, weight="bold")
    box(ax, c2, head_y, cw, 0.16,
        "nicht-stationär\n(dauerhafte Verteilungsänderung)\n→ Schiene C",
        C_NONSTAT, fontsize=10, weight="bold")
    ax.text(x0 + xw / 2, head_y + 0.08,
            "Grundtyp\n(Chandola u. a. 2009)",
            ha="center", va="center", fontsize=10, weight="bold")

    # Zeilen-Labels (Achse 1: Grundtypen)
    labels = ["Punktanomalie", "kontextuelle\nAnomalie", "kollektive\nAnomalie"]
    for y, lab in zip(rows_y, labels):
        box(ax, x0, y, xw, rh, lab, C_HEAD, fontsize=10)

    # Zellen stationär
    box(ax, c1, rows_y[0], cw, rh, "Spike · Drop\n(einzelner Wert auffällig)",
        C_STAT)
    box(ax, c1, rows_y[1], cw, rh,
        "Leckage (kurz)\n(Niveau nur im Kontext auffällig)", C_STAT)
    box(ax, c1, rows_y[2], cw, rh,
        "Plateau\n(eingefrorene Wertefolge)", C_STAT)

    # Zellen nicht-stationär: Punkt entfällt; Regimewechsel spannt
    # kontextuell + kollektiv (Drift/Strukturbruch wirken über beide).
    box(ax, c2, rows_y[0], cw, rh,
        "—\n(per Definition zeitlich ausgedehnt)", "#ffffff", fontsize=9)
    box(ax, c2, rows_y[2], cw, rows_y[1] - rows_y[2] + rh,
        "Drift (schleichender Regimewechsel)\n"
        "Strukturbruch (abrupter Regimewechsel)\n"
        "dauerhafte Leckage", C_NONSTAT)

    ax.text(0.5, 0.006,
            "Schiene A: Rohsignal (Feature-Raum) · Schiene B: MSTL-Residuum · "
            "Schiene C: MSTL-Trend",
            ha="center", va="bottom", fontsize=9, style="italic",
            color="#444444")
    ax.set_title("Anomalietypen: Grundtypen × Stationarität und Zuordnung "
                 "zu den Detektions-Schienen", fontsize=11, pad=14)

    fig.tight_layout()
    out = fig_dir / "anomalietypen_schema.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
