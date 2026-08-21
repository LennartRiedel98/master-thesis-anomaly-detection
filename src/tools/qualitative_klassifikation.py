"""Protokoll-Klassifikation der Top-50-Listen (qualitative Validierung).

Wendet das in der Thesis (Abschnitt 3.5.2) festgelegte Protokoll auf die
Stage-10-Top-50-Listen an: Jede der 50 höchstbewerteten Test-Stunden eines
Jobs wird gegen die injizierte Ground-Truth beider Schienen-Gruppen und die
realen Fehler-Flags klassifiziert:

  TP:<typ>            - Stunde liegt in einem injizierten Ereignis der
                        eigenen Schiene (raw/residual: stationär, trend:
                        nicht-stationär)
  andere-Schiene:<typ> - Stunde liegt in einem injizierten Ereignis der
                        jeweils anderen Schiene (kein Fehlalarm im engeren
                        Sinn, aber Zuständigkeits-Übergriff)
  realer-Datenfehler  - real geflaggte Sensorfehler-Stunde (Zwei-Stufen-
                        Filter), kein injiziertes Ereignis
  no_data             - Stunde ohne Daten
  Fehlalarm/offen     - keine der obigen Kategorien

Liefert die in Kap. 4.6 (Fallstudien A-C, Champion-Inspektion) berichteten
Zahlen. Ausgabe: reports/qualitative_klassifikation.csv (eine Zeile je
Job x Kategorie) + Konsolen-Zusammenfassung der Fallstudien-Jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

# Jobs, die in Kap. 4.6 explizit berichtet werden
HIGHLIGHT = [
    "raw_wmz_2_constancy",       # Fallstudie A
    "residual_wmz_1_lof",        # Fallstudie B
    "trend_wmz_1_pelt",          # Fallstudie C
    "raw_wmz_1_iforest",         # Champion, sauberster Zähler
    "raw_wmz_3_iforest",         # Champion, fehlerbehafteter Zähler
]


def classify_row(row: pd.Series, gt_col: str, other_col: str) -> str:
    """Eine der 50 hoechstbewerteten Stunden einer Kategorie zuordnen.

    Die Reihenfolge der Pruefungen ist die Aussage: Zuerst gilt ein Treffer
    auf der eigenen Schiene (TP), dann ein Treffer auf der anderen Schiene
    (formal ein Fehlalarm, inhaltlich aber eine echte Anomalie), dann ein
    realer, bereits bekannter Datenfehler, dann eine Stunde ohne Daten. Was
    uebrig bleibt, ist ein echter Fehlalarm.

    Diese Abstufung ist der Grund, warum die qualitative Auswertung ein
    anderes Bild ergibt als die reine Precision: Ein grosser Teil der
    formalen Fehlalarme trifft reale Auffaelligkeiten.
    """
    if pd.notna(row[gt_col]):
        return f"TP:{row[gt_col]}"
    if pd.notna(row[other_col]):
        return f"andere-Schiene:{row[other_col]}"
    if row["known_sensor_issue"]:
        return "realer-Datenfehler"
    if row["no_data"]:
        return "no_data"
    return "Fehlalarm/offen"


def main() -> None:
    """Das Klassifikationsprotokoll auf die Top-50-Listen aller Jobs anwenden."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    args = p.parse_args()
    rep_dir = ROOT / "outputs" / args.dataset / "reports"

    rows = []
    for f in sorted(rep_dir.glob("stage10_qualitative_*.csv")):
        job = f.stem.replace("stage10_qualitative_", "")
        parts = job.split("_")
        variant, meter, model = parts[0], "_".join(parts[1:3]), "_".join(parts[3:])
        gt_col = "kind_stat" if variant in ("raw", "residual") else "kind_nonstat"
        other_col = "kind_nonstat" if gt_col == "kind_stat" else "kind_stat"

        df = pd.read_csv(f)
        # Binäre Scorer (PELT): die Top-50-Liste ist mit Null-Score-Stunden
        # aufgefüllt, sobald weniger als 50 Stunden markiert sind - gewertet
        # werden nur die tatsächlich markierten Stunden (score == max > 0).
        if df["score"].nunique() <= 2 and df["score"].max() > 0:
            df = df[df["score"] == df["score"].max()]
        cats = df.apply(classify_row, axis=1, args=(gt_col, other_col))
        for cat, n in cats.value_counts().items():
            rows.append({"job": job, "variant": variant, "meter": meter,
                         "model": model, "kategorie": cat, "stunden": int(n)})

    out = pd.DataFrame(rows)
    out_path = rep_dir / "qualitative_klassifikation.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out['job'].nunique()} Jobs)")

    for job in HIGHLIGHT:
        sub = out[out["job"] == job]
        if sub.empty:
            print(f"\n{job}: keine Top-50-Liste gefunden")
            continue
        print(f"\n=== {job} ===")
        for _, r in sub.sort_values("stunden", ascending=False).iterrows():
            print(f"  {r['kategorie']:<28} {r['stunden']:>3}")


if __name__ == "__main__":
    main()
