"""Recall-Heatmap (Detektor x Anomalietyp) als Abbildung zu Tabelle 11.

Reines Plotten aus vorhandenen Stage-10-Outputs - kein Training, kein Lauf.
Liest outputs/<ds>/reports/stage10_metrics.csv, filtert die type:<typ>-Strata
und bildet je (Modell, Typ) den Bestwert (max Recall) ueber alle Jobs. Diese
Aggregation reproduziert Tabelle 11 der MA exakt (verifiziert 2026-07-28);
NaN-Zellen entsprechen den "-"-Zellen (Modell auf seiner Schiene fuer den Typ
nicht zustaendig).

Darstellung (Dataviz-Vorgaben):
  - sequenzielle Ein-Ton-Skala hell->dunkel (Recall = Magnitude), kein Regenbogen
  - nicht-zutreffende Zellen schraffiert mit "-" (klar von Recall 0 getrennt)
  - Wertelabel in jeder Zelle (Identitaet nie nur ueber Farbe)
  - Zeilenreihenfolge zeigt Plateau-Luecke (iforest 0) und Schliessung
    (constancy 1,0) direkt untereinander.

Aufruf:  python src/tools/plot_recall_heatmap.py [--dataset demo_synthetic]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reihenfolge und Beschriftungen exakt wie in Tabelle 11 der MA
MODELS = ["zscore", "lof", "iforest", "constancy", "pelt", "lstm_ae"]
MODEL_LABELS = ["Z-Score", "LOF", "IForest", "Konstanz", "PELT", "LSTM-AE"]
TYPES = ["spike", "drop", "plateau", "leakage", "drift", "structural_break"]
TYPE_LABELS = ["Spike", "Drop", "Plateau", "Leckage", "Drift", "Strukturbruch"]


def build_matrix(csv_path: Path) -> pd.DataFrame:
    """Recall-Matrix (Modell x Anomalietyp) aus den Stage-10-Metriken bilden.

    Gefiltert werden die nach Anomalietyp aufgeschluesselten Zeilen; je
    Kombination wird der Bestwert ueber alle Jobs genommen. Diese Aggregation
    entspricht genau der Tabelle in der Arbeit. Leere Zellen bedeuten, dass
    das Modell auf seiner Schiene fuer diesen Typ nicht zustaendig ist.
    """
    df = pd.read_csv(csv_path)
    t = df[df["stratum"].str.startswith("type:")].copy()
    t["typ"] = t["stratum"].str.replace("type:", "", regex=False)
    # Bestwert je (Modell, Typ) ueber alle Jobs (variant x wmz)
    piv = t.groupby(["model", "typ"])["recall"].max().unstack()
    return piv.reindex(index=MODELS, columns=TYPES)


def plot(mat: pd.DataFrame, out_path: Path) -> None:
    """Die Recall-Matrix als Heatmap zeichnen und speichern.

    Nicht zustaendige Kombinationen bleiben maskiert statt als 0 zu
    erscheinen - eine 0 wuerde 'erkannt nichts' bedeuten und damit etwas
    anderes als 'gar nicht erst angetreten'.
    """
    data = mat.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("Blues").copy()
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto")

    # Achsen
    ax.set_xticks(range(n_cols), labels=TYPE_LABELS)
    ax.set_yticks(range(n_rows), labels=MODEL_LABELS)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xlabel("Anomalietyp")
    ax.set_ylabel("Detektor")

    # feines Gitter zwischen den Zellen (recessiv)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Zellinhalt: Wert oder schraffiertes "-" fuer nicht zutreffend
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True, facecolor="#eceff1",
                    edgecolor="white", hatch="////", linewidth=0))
                ax.text(j, i, "–", ha="center", va="center",
                        color="#90a4ae", fontsize=11)
            else:
                txt = f"{val:.2f}".replace(".", ",")
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val >= 0.55 else "#102027",
                        fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Recall")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Heatmap aus vorhandenen Stage-10-Ergebnissen erzeugen.

    Rein darstellend - kein Training, kein Lauf.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="demo_synthetic")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "outputs" / args.dataset / "reports" / "stage10_metrics.csv"
    out_path = root / "outputs" / args.dataset / "figures" / "recall_heatmap.png"
    mat = build_matrix(csv_path)
    print("Recall-Matrix (Bestwert je Modell x Typ):")
    print(mat.round(2).to_string())
    plot(mat, out_path)
    print(f"\ngespeichert: {out_path}")


if __name__ == "__main__":
    main()
