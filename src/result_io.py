"""Nicht-destruktives Schreiben der Stage-8/10-Ergebnisdateien.

Stage 8 und Stage 10 koennen mit ``--variants/--wmz/--models`` auf
Teilmengen laufen (z. B. erst die klassischen Modelle, spaeter der teure
LSTM-AE getrennt). Damit ein Teillauf die Ergebnisse anderer Modelle nicht
ueberschreibt, **mergen** diese Helfer neue Resultate in die bestehenden
Dateien: bestehende Eintraege der in diesem Lauf bearbeiteten Schluessel
werden ersetzt, alle anderen bleiben erhalten.

Mit ``fresh=True`` wird die Datei stattdessen komplett neu geschrieben
(sauberer Rebuild von Grund auf).

Damit erledigt sich der frueher noetige manuelle Snapshot-/Merge-Tanz
(das Hilfstool tools/merge_results.py wurde damit obsolet und entfernt).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def merge_best_hparams(base: dict, new: dict) -> dict:
    """Tief-Merge auf ``[variant][wmz][model]``-Ebene; ``new`` gewinnt.

    Eintraege, die nur in ``base`` stehen (= aus einem frueheren Lauf),
    bleiben erhalten; kollidierende Schluessel werden durch ``new``
    (= aktueller Lauf) ersetzt.
    """
    # Tiefe Kopie via JSON-Roundtrip, damit base nicht mutiert wird.
    out = json.loads(json.dumps(base))
    for variant, wmz_map in new.items():
        for wmz, model_map in wmz_map.items():
            for model_name, payload in model_map.items():
                out.setdefault(variant, {}).setdefault(wmz, {})[model_name] = payload
    return out


def write_best_hparams(path: Path, new: dict, fresh: bool = False) -> dict:
    """Schreibt best_hparams.json - merged in Bestehendes, ausser ``fresh``."""
    if not fresh and path.is_file():
        with open(path) as fh:
            new = merge_best_hparams(json.load(fh), new)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(new, fh, indent=2)
    return new


def write_rows_csv(path: Path, new_df: pd.DataFrame, key_cols: list[str],
                   fresh: bool = False) -> pd.DataFrame:
    """Schreibt eine Ergebnis-CSV nicht-destruktiv.

    Bei vorhandener Datei (und nicht ``fresh``) werden aus der alten Tabelle
    alle Zeilen entfernt, deren ``key_cols``-Kombination im aktuellen Lauf
    ebenfalls vorkommt (= wird neu vermessen); danach werden die neuen
    Zeilen angehaengt. Eintraege zu Schluesseln, die in diesem Lauf nicht
    vorkamen, bleiben unveraendert erhalten.
    """
    if not fresh and path.is_file():
        old = pd.read_csv(path)
        if set(key_cols).issubset(old.columns):
            # Schluessel-Kombinationen des aktuellen Laufs.
            done = set(map(tuple, new_df[key_cols].drop_duplicates()
                           .to_numpy().tolist()))
            keep = ~old[key_cols].apply(lambda r: tuple(r) in done, axis=1)
            new_df = pd.concat([old[keep], new_df], ignore_index=True, sort=False)
        else:
            # Alte Datei hat ein anderes Schema - sicherheitshalber anhaengen.
            new_df = pd.concat([old, new_df], ignore_index=True, sort=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(path, index=False)
    return new_df
