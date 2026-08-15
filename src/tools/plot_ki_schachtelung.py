"""Konzept-Diagramm: Schachtelung KI > Maschinelles Lernen > Deep Learning.

Reines Konzept-Schaubild (keine Daten) zu Abschnitt 2.1 der MA: ordnet die
sechs Verfahren der Arbeit in die uebliche Begriffsschachtelung ein
(KI umfasst ML, ML umfasst DL; Himeur et al., 2021). Verfahren ohne
Lernkomponente (MSTL, Z-Score, PELT, Konstanz-Detektor) stehen bewusst
ausserhalb der KI-Menge.

Darstellung: Ein-Ton-Blau hell->dunkel fuer die Schachtelungstiefe,
Identitaet immer zusaetzlich ueber Textlabel (nie nur Farbe).

Aufruf:  python src/tools/plot_ki_schachtelung.py
Ausgabe: docs/ki_schachtelung.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INK = "#102027"
MUTED = "#546e7a"


def main() -> None:
    """Das Schaubild zur Begriffsschachtelung zeichnen und nach docs/ schreiben.

    Ordnet die Verfahren dieser Arbeit in die uebliche Schachtelung ein:
    Kuenstliche Intelligenz umfasst maschinelles Lernen, dieses wiederum Deep
    Learning. Verfahren ohne Lernkomponente (MSTL, Z-Score, PELT, der
    Konstanz-Detektor) stehen bewusst ausserhalb der KI-Menge.

    Achtung: Das Skript hat keine Argumentbehandlung - jeder Aufruf zeichnet
    und ueberschreibt ``docs/ki_schachtelung.png``, auch ein Aufruf mit
    ``--help``.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(-2.1, 5.4)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")

    # Drei unten buendige Kreise: aussen KI, mitte ML, innen DL
    circles = [
        (0.00, 1.55, "#deebf7"),   # KI
        (-0.50, 1.05, "#9ecae1"),  # ML
        (-0.90, 0.65, "#3182bd"),  # DL
    ]
    for cy_off, r, color in circles:
        ax.add_patch(mpatches.Circle((0, cy_off), r, facecolor=color,
                                     edgecolor="white", linewidth=2, zorder=2))

    ax.text(0, 1.22, "Künstliche Intelligenz", ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=3)
    ax.text(0, 0.32, "Maschinelles Lernen", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=3)
    ax.text(0, 0.05, "LOF, Isolation Forest", ha="center", va="center",
            fontsize=9.5, color=INK, zorder=3)
    ax.text(0, -0.70, "Deep Learning", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=3)
    ax.text(0, -0.95, "LSTM-Autoencoder", ha="center", va="center",
            fontsize=8.5, color="white", zorder=3)

    # Lesart-Annotationen rechts an den Kreisen
    ax.annotate("„KI im weiten Sinn“:\nschließt ML ein",
                xy=(0.93, 0.05), xytext=(2.1, 0.75),
                fontsize=9, color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.annotate("„KI im engen Sinn“:\nnur Deep Learning",
                xy=(0.45, -0.85), xytext=(2.1, -0.35),
                fontsize=9, color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

    # Kasten ausserhalb der KI-Menge: klassische Verfahren ohne Lernkomponente
    box = mpatches.FancyBboxPatch((2.05, -1.62), 3.1, 0.92,
                                  boxstyle="round,pad=0.06",
                                  facecolor="#eceff1", edgecolor="#90a4ae",
                                  linewidth=1)
    ax.add_patch(box)
    ax.text(3.6, -0.92, "Klassische Statistik / Regeln\n(ohne Lernkomponente)",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=INK)
    ax.text(3.6, -1.38, "MSTL-Zerlegung, Z-Score,\nPELT, Konstanz-Detektor",
            ha="center", va="center", fontsize=9.5, color=INK)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[2] / "docs" / "ki_schachtelung.png"
    # Zielordner anlegen: docs/ ist kein Ablageort fuer erzeugte Bilder mehr
    # (die Abbildung steht in der Arbeit), der Ordner kann also fehlen.
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {out}")


if __name__ == "__main__":
    main()
