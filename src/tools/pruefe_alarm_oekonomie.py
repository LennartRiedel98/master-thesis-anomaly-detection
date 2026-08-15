"""Prueft die Alarm-Oekonomie-Zahlen aus MA-Abschnitt 4.5 gegen die Metriken.

Hintergrund: Jede Zahl im Ergebnisteil der Arbeit ist den Weg
``stage10_metrics.csv`` -> ``ergebnisbericht.md`` -> Arbeit gegangen und dabei
mindestens einmal gegengelesen worden. Die Angaben zur Alarm-Oekonomie
(Absatz "Zahl markierter Stunden") sind direkt in die Arbeit gewandert; die
Spalte ``n_pred`` wurde nie in den Bericht uebernommen. Dieses Skript holt das
nach: Es liest die Metrik-CSV und stellt Soll und Ist nebeneinander.

Wichtig zur Spalte ``n_pred``: Sie bedeutet je nach Stratum etwas anderes.
In den ``overall``-Zeilen ist es die **Zahl markierter Stunden** - genau die
Groesse, die die Arbeit nennt. In den ``type:*``-Zeilen dagegen die Zahl
erkannter Ereignisse. Geprueft wird deshalb ausschliesslich ``overall``.

Aufruf::

    python src/tools/pruefe_alarm_oekonomie.py
    python src/tools/pruefe_alarm_oekonomie.py --csv <pfad>   # Snapshot pruefen

Ausgabe: nur Konsole. Exit-Code 1, wenn eine Zahl abweicht oder fehlt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "demo_synthetic"

# Was Abschnitt 4.5 behauptet: (Variante, Zaehler, Modell) -> Sollwerte.
# Quelle: Abschnitt 4.5 der Arbeit, Absatz "Betrieblich ebenso relevant
# wie die Trefferquote..."
SOLL = [
    ("raw",   "wmz_1", "iforest",   {"n_pred": 235,   "precision": 0.876}),
    ("trend", "wmz_1", "pelt",      {"n_pred": 11,    "precision": 0.997}),
    ("trend", "wmz_1", "lstm_ae",   {"n_pred": 1088,  "precision": 0.712}),
    ("raw",   "wmz_1", "constancy", {"precision": 0.924, "recall": 0.176}),
]

# Toleranz: Die Arbeit rundet auf drei Nachkommastellen.
TOL = 0.0006

RUECKFALL = """
Die Metrik-CSV wurde nicht gefunden. Vier Ebenen, in dieser Reihenfolge:

  1. WEITERSUCHEN, bevor neu gerechnet wird. Die Datei war nie im Git
     (outputs/ ist gitignored). Es gibt aber Snapshot-Kopien im selben
     Ordner - stage10_metrics_<suffix>.csv mit Suffixen wie _classic,
     _before_gpu_nightrun, _gridwh_cpu, _gridwh_gpu_roc_auc,
     _option_e_pr_auc_gpu, _pre_seedfix, _seedstd, _option_e_f1_gpu -
     dazu alles unter outputs/_archiv/. Fuer die Klassik-Zahlen taugt
     jeder dieser Staende. Mit --csv <pfad> direkt pruefen.
     Ausserdem: Der Ordner existiert auf zwei Rechnern (Mac + RTX-Laptop).

  2. KLASSIK NEU RECHNEN - exakt und in Minuten, weil deterministisch:
       python src/stage1_load.py ... bis ... src/stage6_normalize.py
       python src/stage9_inject.py
       python src/stage8_hpo.py       --models iforest constancy pelt
       python src/stage10_evaluate.py --models iforest constancy pelt
     Das bringt 235, elf, 0,924 und 0,176 exakt zurueck.

  3. DIE 1.088 IST DER SONDERFALL. Sie stammt vom LSTM-AE, der auf GPU
     nicht bit-reproduzierbar ist (das sagt Limitation L6). Ein Neulauf
     liefert einen aehnlichen, nicht denselben Wert.

  4. LETZTE EBENE: die Zahl aus dem Satz nehmen. Das Argument des Absatzes
     ist das Verhaeltnis ("ein Vielfaches an zu pruefenden Alarmen"), nicht
     der Absolutwert. Nur wenn 3. ausscheidet - eine gemessene Zahl zu
     berichten ist richtig, solange sie stimmt.

"""


def finde_csv(dataset: str, vorgabe: Path | None) -> Path | None:
    """Metrik-CSV suchen: erst die Vorgabe, dann der kanonische Pfad."""
    if vorgabe is not None:
        return vorgabe if vorgabe.is_file() else None
    pfad = ROOT / "outputs" / dataset / "reports" / "stage10_metrics.csv"
    return pfad if pfad.is_file() else None


def zeige_snapshots(dataset: str) -> None:
    """Vorhandene Snapshot-Kopien auflisten, falls es welche gibt."""
    ordner = ROOT / "outputs" / dataset / "reports"
    if not ordner.is_dir():
        print(f"  (Ordner {ordner} existiert nicht)")
        return
    kopien = sorted(ordner.glob("stage10_metrics*.csv"))
    if kopien:
        print("  Gefundene Kopien im reports-Ordner:")
        for k in kopien:
            print(f"    {k.name}")
    else:
        print("  Keine stage10_metrics*.csv im reports-Ordner.")


def pruefe_zeile(zeile: pd.Series, erwartet: dict[str, float]) -> list[str]:
    """Eine Job-Zeile gegen die Sollwerte halten; liefert Abweichungstexte."""
    abweichungen = []
    for feld, soll in erwartet.items():
        if feld not in zeile.index or pd.isna(zeile[feld]):
            abweichungen.append(f"{feld}: Spalte fehlt oder leer")
            continue
        ist = float(zeile[feld])
        passt = abs(ist - soll) < (0.5 if feld == "n_pred" else TOL)
        zeichen = "OK  " if passt else "ABW "
        stil = f"{ist:.0f}" if feld == "n_pred" else f"{ist:.3f}"
        soll_stil = f"{soll:.0f}" if feld == "n_pred" else f"{soll:.3f}"
        print(f"      {zeichen} {feld:10s} Arbeit: {soll_stil:>8s}   CSV: {stil:>8s}")
        if not passt:
            abweichungen.append(f"{feld}: Arbeit {soll_stil}, CSV {stil}")
    return abweichungen


def main() -> int:
    """Alle vier Jobs pruefen und ein Gesamturteil ausgeben."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--csv", type=Path, default=None,
                   help="Andere Metrik-CSV pruefen (z. B. einen Snapshot)")
    args = p.parse_args()

    pfad = finde_csv(args.dataset, args.csv)
    if pfad is None:
        print("FEHLER: Metrik-CSV nicht gefunden.\n")
        zeige_snapshots(args.dataset)
        print(RUECKFALL)
        return 1

    m = pd.read_csv(pfad)
    print(f"Quelle: {pfad}")
    print(f"        {len(m)} Zeilen, {m['model'].nunique()} Modelle\n")

    ov = m[m["stratum"] == "overall"]
    alle_abweichungen: list[str] = []
    fehlend: list[str] = []

    for variante, wmz, modell, erwartet in SOLL:
        job = f"{variante}/{wmz}/{modell}"
        treffer = ov[(ov["variant"] == variante) & (ov["wmz"] == wmz)
                     & (ov["model"] == modell)]
        print(f"  {job}")
        if treffer.empty:
            print("      FEHLT - dieser Job steht nicht in der CSV")
            fehlend.append(job)
            continue
        alle_abweichungen += [f"{job} {a}" for a in pruefe_zeile(treffer.iloc[0], erwartet)]

    # PELT laeuft auf allen drei Zaehlern - die Arbeit nennt "elf Stunden"
    # ohne Zaehler. Zur Sicherheit alle drei zeigen.
    print("\n  Kontrolle: PELT auf allen Zaehlern (Arbeit nennt 'elf Stunden')")
    for _, z in ov[ov["model"] == "pelt"].iterrows():
        print(f"      {z['variant']}/{z['wmz']}: n_pred={z['n_pred']:.0f} "
              f"precision={z['precision']:.3f}")

    print()
    if fehlend:
        print(f"{len(fehlend)} Job(s) fehlen in der CSV: {', '.join(fehlend)}")
    if alle_abweichungen:
        print(f"{len(alle_abweichungen)} ABWEICHUNG(EN) - Absatz 4.5 korrigieren:")
        for a in alle_abweichungen:
            print(f"  - {a}")
        return 1
    if fehlend:
        return 1
    print("Alle geprueften Zahlen stimmen mit der Arbeit ueberein.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
