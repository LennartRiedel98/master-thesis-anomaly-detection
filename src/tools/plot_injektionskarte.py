"""Visualisiert die synthetische Anomalie-Injektion (Stage 9) als Zeitleiste.

Liest ``stage9_ground_truth.parquet`` und zeichnet je WMZ eine Gantt-artige
Karte: pro Anomalietyp eine Spur (Lane), jedes injizierte Event als Balken
von Start bis Ende. So ist auf einen Blick sichtbar, *wann*, *welcher Typ*,
*welche Intensitaet* und *wie lange* in das Test-Fenster injiziert wurde.

Aufruf:
    python src/tools/plot_injektionskarte.py --dataset demo_synthetic
    python src/tools/plot_injektionskarte.py --docs-copy   # zusaetzlich nach docs/

Die Figure landet immer in ``outputs/<ds>/figures/`` (gitignored). Mit
``--docs-copy`` wird sie zusaetzlich nach ``docs/injektionskarte.png``
gespiegelt - praktisch, um sie ohne Umweg ueber den outputs/-Ordner
griffbereit zu haben. Versioniert ist auch diese Kopie nicht: Die Abbildung
steht in der Arbeit, im Repo steht der Code, der sie erzeugt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"
WMZ_NAMES = ["wmz_1", "wmz_2", "wmz_3"]

# Reihenfolge der Lanes (stationaer zuerst, dann nicht-stationaer) + Farben.
TYPE_ORDER = ["spike", "drop", "plateau", "leakage", "drift", "structural_break"]
TYPE_COLOR = {
    "spike": "#d62728",
    "drop": "#1f77b4",
    "plateau": "#9467bd",
    "leakage": "#ff7f0e",
    "drift": "#2ca02c",
    "structural_break": "#8c564b",
}
# Mindestbreite eines Balkens in Stunden, damit 1-h-Events sichtbar bleiben.
MIN_BAR_HOURS = 8


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--docs-copy", action="store_true",
                   help="Figure zusaetzlich nach docs/injektionskarte.png "
                        "spiegeln (versioniertes Asset).")
    return p.parse_args()


def extract_events(labels: pd.Series) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """Zusammenhaengende Label-Laeufe als (typ, label, start, ende) extrahieren."""
    s = labels.fillna("")
    idx = s.index
    events: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    prev, start = "", None
    for i, v in enumerate(s.values):
        if v != prev:
            if prev != "" and start is not None:
                kind = prev.split("@")[0]
                events.append((kind, prev, idx[start], idx[i - 1]))
            start = i if v != "" else None
            prev = v
    if prev != "" and start is not None:
        kind = prev.split("@")[0]
        events.append((kind, prev, idx[start], idx[-1]))
    return events


def main() -> None:
    """Die synthetische Injektion als Zeitleiste zeichnen, je Zaehler eine Karte.

    Pro Anomalietyp eine Spur, jedes injizierte Ereignis als Balken von Start
    bis Ende. Damit ist auf einen Blick zu sehen, wann, welcher Typ, mit
    welcher Intensitaet und ueber welche Dauer ins Test-Fenster injiziert
    wurde - und dass die Ereignisse sich nicht ueberlappen.
    """
    args = parse_args()
    gt_path = ROOT / "outputs" / args.dataset / "parquet" / "stage9_ground_truth.parquet"
    fig_dir = ROOT / "outputs" / args.dataset / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(gt_path)

    # Nur das Test-Fenster zeigen (dort wird injiziert).
    all_events: dict[str, list] = {}
    t_min, t_max, total = None, None, 0
    for wmz in WMZ_NAMES:
        evs = []
        for lab_col in (f"gt_stat_label_{wmz}", f"gt_nonstat_label_{wmz}"):
            evs.extend(extract_events(df[lab_col]))
        all_events[wmz] = evs
        total += len(evs)
        for _, _, a, b in evs:
            t_min = a if t_min is None else min(t_min, a)
            t_max = b if t_max is None else max(t_max, b)

    min_bar = pd.Timedelta(hours=MIN_BAR_HOURS)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for ax, wmz in zip(axes, WMZ_NAMES):
        for kind, label, a, b in all_events[wmz]:
            y = TYPE_ORDER.index(kind)
            width = max(b - a, min_bar)
            ax.barh(y, width=width, left=a, height=0.6,
                    color=TYPE_COLOR[kind], alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(TYPE_ORDER)))
        ax.set_yticklabels(TYPE_ORDER, fontsize=9)
        ax.set_ylim(-0.6, len(TYPE_ORDER) - 0.4)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_ylabel(wmz, fontsize=11, fontweight="bold")
        ax.axhline(3.5, color="grey", linestyle="--", linewidth=0.7)  # stat | nonstat

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].set_xlim(t_min - pd.Timedelta(days=4), t_max + pd.Timedelta(days=4))

    handles = [mpatches.Patch(color=TYPE_COLOR[t], label=t) for t in TYPE_ORDER]
    axes[0].legend(handles=handles, ncol=6, loc="upper center",
                   bbox_to_anchor=(0.5, 1.32), fontsize=9, frameon=False)
    fig.suptitle(
        f"Injektions-Karte: {total} synthetische Anomalien im Test-Fenster "
        f"(4 Events je Typ×Intensität × 3 WMZ)",
        fontsize=13, y=0.99)
    fig.text(0.5, 0.02, "gestrichelte Linie = Trennung stationär (oben) / "
             "nicht-stationär (unten); Balken min. 8 h breit dargestellt",
             ha="center", fontsize=8, color="grey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = fig_dir / "injektionskarte.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}  ({total} Events)")

    # Optional als versioniertes Asset nach docs/ spiegeln (siehe Docstring).
    if args.docs_copy:
        import shutil
        docs_out = ROOT / "docs" / "injektionskarte.png"
        docs_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, docs_out)
        print(f"-> {docs_out}  (versionierte Kopie)")


if __name__ == "__main__":
    main()
