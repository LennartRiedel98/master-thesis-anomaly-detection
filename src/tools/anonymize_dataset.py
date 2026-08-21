"""Einen Zaehler-Datensatz in eine anonymisierte Fassung ueberfuehren.

Aufruf::

    python src/tools/anonymize_dataset.py --quelle data/<original> \
                                          --ziel   data/gebaeude_a

Ersetzt werden ausschliesslich **identifizierende Metadaten**:

===========================  ==================================================
Dateiname                    ``logging_heat-energy_<kennung>_<zeitraum>.csv``
                             -> ``logging_heat-energy_gebaeude-a_<zeitraum>.csv``
Kopfzeile der Zaehler-CSVs   die Zaehler-IDs in Kopfreihenfolge
                             -> ``10000001``, ``10000002``, ...
Wetterdatei                  Name -> ``open-meteo-station.csv``;
                             Breite und Laenge -> Stadtmitte
===========================  ==================================================

**Messwerte, Zeitstempel und Spaltenreihenfolge bleiben unangetastet.** Das ist
Absicht: nur so bleiben alle in der Arbeit berichteten Zahlen auf der
anonymisierten Fassung nachrechenbar. Weil die Reihenfolge im Dateikopf
erhalten bleibt, liefert ``stage1_load.py`` unveraendert ``wmz_1``, ``wmz_2``
und ``wmz_3``.

Das Skript enthaelt **keine** realen Kennungen: es liest sie aus der Kopfzeile
und ersetzt das n-te gefundene Geraet durch ``1000000n``. Es kann deshalb
gefahrlos mit veroeffentlicht werden.

Gearbeitet wird binaer - ausser der ersten Zeile (Zaehler) bzw. der zweiten
Zeile (Wetter) ist die Ausgabe byte-identisch mit der Eingabe.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WMZ_GLOB = "logging_heat-energy_*.csv"
WEATHER_GLOB = "open-meteo-*.csv"

# Zielkennungen. Die Nummernreihe ist frei gewaehlt und ohne Bezug zu realen
# Geraeten; sie stimmt mit src/tools/make_synthetic_data.py ueberein, damit
# echte und synthetische Fassung dasselbe Format zeigen.
GEBAEUDE_ALIAS = "gebaeude-a"
ZAEHLER_BASIS = 10000001
WETTER_NAME = "open-meteo-station.csv"
# Grob auf die Stadtmitte gesetzt: das Format der Kopfzeile bleibt erhalten,
# der reale Standort verschwindet. Die Hoehe ist unkritisch und bleibt stehen.
KOORDINATEN = ("52.52", "13.405")

# logging_heat-energy_<kennung>_<zeitraum>.csv - Gruppe 1 ist die Kennung.
DATEINAME_MUSTER = re.compile(r"^(?:logging_heat-energy_)([^_]+)(_.*\.csv)$")


def md5_ab_zeile(pfad: Path, kopfzeilen: int) -> str:
    """MD5 ueber alles ab ``kopfzeilen + 1`` - den Teil, der gleich bleiben muss."""
    h = hashlib.md5()
    with open(pfad, "rb") as fh:
        for _ in range(kopfzeilen):
            fh.readline()
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def zaehler_ersetzungen(kopfzeile: bytes) -> dict[bytes, bytes]:
    """Zaehler-IDs in Kopfreihenfolge auf 10000001, 10000002, ... abbilden.

    Erwartet Spalten der Form ``<ID> / <Einheit>``; jede ID kommt zweimal vor
    (MWh und kW), gezaehlt wird ihr erstes Auftreten - dieselbe Konvention wie
    in ``stage1_load.meter_map_from_header``.
    """
    ersetzungen: dict[bytes, bytes] = {}
    for spalte in kopfzeile.split(b";"):
        kennung, trenner, _einheit = spalte.partition(b" / ")
        if not trenner:
            continue
        kennung = kennung.strip()
        if kennung and kennung not in ersetzungen:
            ersetzungen[kennung] = str(ZAEHLER_BASIS + len(ersetzungen)).encode()
    return ersetzungen


def kopiere_mit_kopfersatz(quelle: Path, ziel: Path, zeilennr: int,
                           ersetzungen: dict[bytes, bytes]) -> int:
    """Datei kopieren, dabei genau eine Zeile ersetzen. Gibt Trefferzahl zurueck."""
    treffer = 0
    with open(quelle, "rb") as ein, open(ziel, "wb") as aus:
        for _ in range(zeilennr - 1):
            aus.write(ein.readline())
        zeile = ein.readline()
        for alt, neu in ersetzungen.items():
            if alt in zeile:
                zeile = zeile.replace(alt, neu)
                treffer += 1
        aus.write(zeile)
        shutil.copyfileobj(ein, aus, length=1 << 20)
    return treffer


def anonymisiere_zaehlerdatei(quelle: Path, zielordner: Path) -> list[str]:
    """Eine Zaehler-CSV anonymisiert ablegen. Gibt gefundene Maengel zurueck."""
    maengel: list[str] = []
    treffer_muster = DATEINAME_MUSTER.match(quelle.name)
    if treffer_muster is None:
        return [f"{quelle.name}: Dateiname folgt nicht dem erwarteten Muster"]
    ziel = zielordner / f"logging_heat-energy_{GEBAEUDE_ALIAS}{treffer_muster.group(2)}"

    with open(quelle, "rb") as fh:
        ersetzungen = zaehler_ersetzungen(fh.readline())
    if not ersetzungen:
        return [f"{quelle.name}: keine Zaehler-IDs in der Kopfzeile gefunden"]

    ersetzt = kopiere_mit_kopfersatz(quelle, ziel, 1, ersetzungen)
    if ersetzt != len(ersetzungen):
        maengel.append(f"{ziel.name}: {ersetzt} von {len(ersetzungen)} IDs ersetzt")
    if md5_ab_zeile(quelle, 1) != md5_ab_zeile(ziel, 1):
        maengel.append(f"{ziel.name}: Messwerte weichen ab")

    print(f"  {quelle.name}\n    -> {ziel.name}  "
          f"({len(ersetzungen)} Zaehler ersetzt, {ziel.stat().st_size:,} Bytes, "
          f"Messwerte unveraendert)")
    return maengel


def anonymisiere_wetterdatei(quelle: Path, zielordner: Path) -> list[str]:
    """Die Open-Meteo-CSV anonymisiert ablegen. Gibt gefundene Maengel zurueck."""
    maengel: list[str] = []
    ziel = zielordner / WETTER_NAME
    with open(quelle, "rb") as fh:
        fh.readline()                      # Feldnamen der Metadatenzeile
        felder = fh.readline().rstrip(b"\r\n").split(b",")
    if len(felder) < 3:
        return [f"{quelle.name}: unerwarteter Metadatenkopf"]

    ersetzungen = {felder[0]: KOORDINATEN[0].encode(),
                   felder[1]: KOORDINATEN[1].encode()}
    ersetzt = kopiere_mit_kopfersatz(quelle, ziel, 2, ersetzungen)
    if ersetzt != len(ersetzungen):
        maengel.append(f"{ziel.name}: {ersetzt} von 2 Koordinaten ersetzt")
    if md5_ab_zeile(quelle, 2) != md5_ab_zeile(ziel, 2):
        maengel.append(f"{ziel.name}: Messwerte weichen ab")

    print(f"  {quelle.name}\n    -> {ziel.name}  "
          f"(Koordinaten -> {KOORDINATEN[0]}/{KOORDINATEN[1]}, "
          f"{ziel.stat().st_size:,} Bytes, Messwerte unveraendert)")
    return maengel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quelle", required=True, type=Path,
                   help="Ordner mit dem Originaldatensatz unter data/")
    p.add_argument("--ziel", required=True, type=Path,
                   help="Zielordner der anonymisierten Fassung, z. B. data/gebaeude_a")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    quelle = args.quelle if args.quelle.is_absolute() else ROOT / args.quelle
    ziel = args.ziel if args.ziel.is_absolute() else ROOT / args.ziel

    if not quelle.is_dir():
        print(f"FEHLER: Quellordner fehlt: {quelle}")
        return 1
    zaehlerdateien = sorted(quelle.glob(WMZ_GLOB))
    wetterdateien = sorted(quelle.glob(WEATHER_GLOB))
    if not zaehlerdateien:
        print(f"FEHLER: keine Datei nach Muster {WMZ_GLOB} in {quelle}")
        return 1

    ziel.mkdir(parents=True, exist_ok=True)
    print(f"Quelle: {quelle}\nZiel  : {ziel}\n")

    maengel: list[str] = []
    for datei in zaehlerdateien:
        maengel += anonymisiere_zaehlerdatei(datei, ziel)
    for datei in wetterdateien:
        maengel += anonymisiere_wetterdatei(datei, ziel)

    if maengel:
        print("\nFEHLER:")
        for m in maengel:
            print(f"  {m}")
        return 1
    print(f"\n{len(zaehlerdateien) + len(wetterdateien)} Dateien anonymisiert, "
          f"keine Beanstandung.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
