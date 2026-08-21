"""Lernkurven aus dem Daten-Sweep (Methodik § 7.8).

Liest ``outputs/<ds>/reports/data_sweep.csv`` und plottet pro Variante
(raw / residual / trend) einen Subplot: Test-F1 (y-Achse, point-adjusted
aus Stage 10) ueber die Anzahl Trainings-Stunden (x-Achse, log-skaliert),
gruppiert nach Modell. Beantwortet visuell die in der Diskussion
zentralen Fragen:

* Ist das Modell **datenlimitiert** (Kurve steigt am rechten Rand noch)?
  Falls ja, ist Pooling/More-Data der vielversprechende Hebel; falls die
  Kurve plateauiert, hilft mehr Trainingsmenge nicht mehr und es liegt
  ein Modell-Class-/Feature-Problem vor.
* **Wo liegt der Klassik-Champion vs. der LSTM-AE** ueber den
  Sweep-Stufen? Liefert das qualitative Bild zum Numerik-Vergleich in
  Ergebnisbericht § 14.8 / § 15.

Konvention: ein Punkt pro (variant, wmz, model, train_rows). Die drei
WMZ pro Modell werden als eigene Linien-Stile gezeichnet (durchgezogen,
gestrichelt, gepunktet), damit der Plot fair vergleichbar bleibt.

Output: ``outputs/<ds>/figures/learning_curves.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

# Farben je Modell - bewusst dieselbe Palette wie in den Stage-10-Plots,
# damit Cross-Plot-Vergleich nicht durch Farbwechsel verwirrt wird.
MODEL_COLORS = {
    "iforest": "#1f77b4",
    "lof":     "#ff7f0e",
    "pelt":    "#2ca02c",
    "lstm_ae": "#d62728",
    "zscore":  "#9467bd",
}

# Linien-Stile je WMZ; identisch ueber alle Subplots.
WMZ_STYLES = {"wmz_1": "-", "wmz_2": "--", "wmz_3": ":"}


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    return p.parse_args()


def main() -> None:
    """Lernkurven aus ``reports/data_sweep.csv`` zeichnen, ein Panel je Schiene.

    Traegt das Test-F1 gegen die Zahl der Trainingsstunden auf (x-Achse
    logarithmisch), gruppiert nach Modell. Die Abbildung beantwortet die
    Frage, ob der schwache Befund des LSTM-AE ein Datenmengen-Problem ist -
    sie zeigt, dass die Kurven flach bleiben.

    Rechnet selbst nichts: Ohne einen vorher gelaufenen Daten-Sweep bleibt
    die Abbildung leer.
    """
    args = parse_args()
    sweep_path = ROOT / "outputs" / args.dataset / "reports" / "data_sweep.csv"
    if not sweep_path.is_file():
        raise SystemExit(f"data_sweep.csv nicht gefunden: {sweep_path}")
    df = pd.read_csv(sweep_path)

    variants = ["raw", "residual", "trend"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, variant in zip(axes, variants):
        sub = df[df["variant"] == variant]
        if sub.empty:
            ax.text(0.5, 0.5, f"(keine Sweep-Daten\nfuer {variant})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{variant}")
            continue
        # Pro (model, wmz)-Linie sortiert nach train_rows.
        for (model, wmz), grp in sub.groupby(["model", "wmz"]):
            grp = grp.sort_values("train_rows")
            ax.plot(grp["train_rows"], grp["f1"],
                    marker="o", linewidth=1.3, markersize=4,
                    color=MODEL_COLORS.get(model, "gray"),
                    linestyle=WMZ_STYLES.get(wmz, "-"),
                    label=f"{model}/{wmz}")
        ax.set_xscale("log")
        ax.set_xlabel("Trainings-Stunden")
        ax.set_title(f"{variant}")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("Test-F1 (point-adjusted)")

    # Eine gemeinsame Legende rechts neben den Subplots - sonst dupliziert
    # sich der Eintrag dreimal. Modell-Farben und WMZ-Linienstile getrennt
    # darstellen, weil sonst bei 12 Linien die Legende explodiert.
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], color=c, linewidth=2, label=m)
                     for m, c in MODEL_COLORS.items()
                     if m in df["model"].unique()]
    wmz_handles = [Line2D([0], [0], color="black", linestyle=s, linewidth=1.3,
                          label=w)
                   for w, s in WMZ_STYLES.items()]
    legend1 = fig.legend(handles=model_handles, loc="center right",
                         bbox_to_anchor=(1.0, 0.65), title="Modell")
    fig.add_artist(legend1)
    fig.legend(handles=wmz_handles, loc="center right",
               bbox_to_anchor=(1.0, 0.35), title="WMZ")

    fig.suptitle("Lernkurven (Test-F1 vs. Trainings-Stunden) je Variante",
                 y=1.02)
    fig.tight_layout(rect=(0, 0, 0.88, 1.0))

    out_path = ROOT / "outputs" / args.dataset / "figures" / "learning_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
