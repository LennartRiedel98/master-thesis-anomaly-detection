"""Misst, wie stark die Detektionsurteile an der gewaehlten Schwelle haengen.

Hintergrund: Die Alarmschwelle ist das 99-%-Quantil der Scores auf dem
*sauberen* Validierungsfenster (``stage10_evaluate.py``). Weil der Split
chronologisch ist, stammt dieses Fenster aus der Heizsaison (November bis
April), bewertet wird aber das Sommerhalbjahr (Mai bis November). Die
Schwelle wird also auf einer anderen Lastverteilung geeicht als der, auf
der sie angewendet wird. Wie sehr das ins Gewicht faellt, haengt daran, wie
viel Reserve die Treffer ueber der Schwelle haben.

Genau diese Reserve steht bereits in Teil B des Fall-Protokolls: Dort traegt
jedes Urteil das Verhaeltnis ``Spitzen-Score im Ereignis / Schwelle``. Ein
Treffer bei 1,02 kippt schon, wenn die Schwelle um zwei Prozent steigt;
einer bei 12 ueberlebt eine Verdopplung. Dieses Skript wertet die Verteilung
aus - **ohne jeden Neulauf**, rein aus dem erzeugten Protokoll.

Nicht beruecksichtigt:

* **PELT** liefert keinen kontinuierlichen Score, sondern 0/1 gegen eine
  feste Schwelle von 0,5. Sein Verhaeltnis ist deshalb immer 2,00 oder 0 und
  sagt ueber Schwellen-Empfindlichkeit nichts aus.
* Der **LSTM-Autoencoder** steht nur in Teil B, wenn das Protokoll mit
  ``--include-lstm`` erzeugt wurde.

Voraussetzung: ``qualitative_protocol.py`` ist gelaufen.
Ausgabe: nur Konsole (die Zahlen stehen in MA-Abschnitt 5.4, L9).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Urteil und Verhaeltnis einer Protokollzelle: "✓ (1.03)" bzw. "✗ (0.52)"
ZELLE = re.compile(r"([✓✗])\s*\(([0-9.]+)\)")

# Binaer scorende Verfahren - ihr Verhaeltnis ist konstruktionsbedingt fix
BINAER = {"pelt"}


def lies_teil_b(pfad: Path) -> dict[str, list[tuple[str, float]]]:
    """Sammelt aus Teil B je Detektor alle (Urteil, Verhaeltnis)-Paare.

    Teil B besteht aus je einer Tabelle pro (Variante x Zaehler); die
    Detektornamen stehen ab der vierten Spalte der Kopfzeile. Ausgewertet
    werden alle Tabellen gemeinsam, denn die Frage nach der Schwellen-
    Empfindlichkeit stellt sich je Detektor, nicht je Schiene.
    """
    daten: dict[str, list[tuple[str, float]]] = defaultdict(list)
    in_teil_b = False
    spalten: list[str] = []

    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        if zeile.startswith("## B."):
            in_teil_b = True
            continue
        if zeile.startswith("## ") and in_teil_b:
            break                      # Teil C beginnt
        if not in_teil_b:
            continue

        if zeile.startswith("### "):
            spalten = []               # neue Tabelle, Kopfzeile abwarten
            continue
        if zeile.startswith("| # |"):
            spalten = [s.strip() for s in zeile.strip("|").split("|")][4:]
            continue
        if not zeile.startswith("|") or not spalten:
            continue

        felder = [s.strip() for s in zeile.strip("|").split("|")]
        for name, feld in zip(spalten, felder[4:]):
            treffer = ZELLE.search(feld)
            if treffer:
                daten[name].append((treffer.group(1), float(treffer.group(2))))
    return daten


def anteil(werte: list[float], grenze: float) -> tuple[int, float]:
    """Anzahl und Anteil der Werte unterhalb einer Grenze."""
    n = sum(1 for v in werte if v < grenze)
    return n, (n / len(werte) if werte else 0.0)


def main() -> int:
    """Reserve-Verteilung je Detektor und im Gesamtbild ausgeben."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--protokoll", type=Path,
                   default=ROOT / "docs" / "qualitative_evaluierung.md",
                   help="Von qualitative_protocol.py erzeugtes Fall-Protokoll")
    args = p.parse_args()

    if not args.protokoll.is_file():
        print(f"FEHLER: {args.protokoll} fehlt - erst qualitative_protocol.py laufen lassen.")
        return 1

    daten = lies_teil_b(args.protokoll)
    if not daten:
        print("FEHLER: In Teil B des Protokolls wurden keine Urteile gefunden.")
        return 1

    bewertbar = {d: z for d, z in daten.items() if d.lower() not in BINAER}

    print(f"Quelle: {args.protokoll.relative_to(ROOT)}\n")
    print("=== Reserve der Treffer ueber der Schwelle ===")
    print(f"{'Detektor':12s} {'Treffer':>8s} {'<1,05':>7s} {'<1,10':>7s} "
          f"{'<1,20':>7s} {'>=2,00':>7s} {'Median':>8s}")
    for det, zeilen in sorted(bewertbar.items()):
        tp = sorted(v for u, v in zeilen if u == "✓")
        if not tp:
            continue
        print(f"{det:12s} {len(tp):8d} {anteil(tp, 1.05)[0]:7d} "
              f"{anteil(tp, 1.10)[0]:7d} {anteil(tp, 1.20)[0]:7d} "
              f"{sum(1 for v in tp if v >= 2.0):7d} {tp[len(tp) // 2]:8.2f}")

    print("\n=== Fehlschlaege dicht unter der Schwelle ===")
    print("(sie wuerden bei einer Absenkung als Erste zu Treffern)")
    for det, zeilen in sorted(bewertbar.items()):
        fn = [v for u, v in zeilen if u == "✗"]
        if fn:
            knapp = sum(1 for v in fn if 0.90 <= v < 1.0)
            print(f"{det:12s} {len(fn):4d} Fehlschlaege, davon {knapp:3d} "
                  f"ab dem 0,90-Fachen der Schwelle")

    alle = [v for zeilen in bewertbar.values() for u, v in zeilen if u == "✓"]
    print(f"\n=== Gesamtbild ({len(alle)} Treffer, ohne binaer scorende Verfahren) ===")
    for grenze in (1.05, 1.10, 1.20):
        n, q = anteil(alle, grenze)
        print(f"  {n:4d} ({q:5.1%}) liegen unter dem {grenze:.2f}-Fachen der Schwelle")
    print("\nLesart: Je groesser diese Anteile, desto staerker haengt das "
          "Ergebnis\nan der Schwellenkalibrierung - und desto mehr wiegt die "
          "saisonale\nVerschiebung zwischen Kalibrierungs- und Bewertungsfenster (L9).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
