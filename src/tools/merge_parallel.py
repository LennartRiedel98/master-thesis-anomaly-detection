"""Merge der parallelen Per-Job-LSTM-Laeufe in das echte Dataset.

Der Parallel-Runner (``run_parallel_lstm.sh``) rechnet jeden der neun
(Variante x WMZ)-Jobs in einem **isolierten** Temp-Dataset
``outputs/_par_<variante>_<wmz>/`` (Parquet/Scaler nur per Symlink
verlinkt), damit es keinen Schreib-Konflikt auf den gemeinsamen
Ergebnisdateien gibt. Dieses Skript fuehrt die Teilergebnisse danach
**einmal, single-threaded** ins echte Dataset zusammen - mit denselben
nicht-destruktiven Merge-Funktionen wie Stage 8/10 (``result_io``), sodass
die klassischen Modell-Eintraege erhalten bleiben und nur die LSTM-Zeilen
aktualisiert werden.

Aufruf:  python src/tools/merge_parallel.py --dataset demo_synthetic
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

# src/ auf den Pfad, damit result_io (liegt in src/) importierbar ist.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_io import merge_best_hparams, write_best_hparams, write_rows_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ["raw", "residual", "trend"]
WMZ = ["wmz_1", "wmz_2", "wmz_3"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--wmz", nargs="+", default=None,
                   help="Nur diese WMZ mergen (Default: alle drei).")
    return p.parse_args()


def main() -> None:
    """Ergebnisse der parallelen Einzellaeufe in den echten Datensatz mergen.

    Der Parallel-Runner rechnet jeden Job in einem eigenen Temp-Datensatz,
    damit die Laeufe nicht gleichzeitig in dieselben Ergebnisdateien schreiben.
    Dieses Skript fuehrt die Teilergebnisse danach einmal und einzeln
    zusammen - mit derselben nicht-destruktiven Merge-Logik wie Stage 8 und 10.
    """
    args = parse_args()
    real = ROOT / "outputs" / args.dataset
    wmz_sel = args.wmz or WMZ
    temp_dirs = [ROOT / "outputs" / f"_par_{v}_{w}"
                 for v in VARIANTS for w in wmz_sel]
    temp_dirs = [t for t in temp_dirs if t.is_dir()]
    print(f"Merge {len(temp_dirs)} Temp-Datasets -> {real}")

    # 1) best_hparams.json: alle Temp-LSTM-Eintraege sammeln, dann ins echte mergen.
    combined: dict = {}
    for t in temp_dirs:
        p = t / "hpo" / "best_hparams.json"
        if p.is_file():
            with open(p) as fh:
                combined = merge_best_hparams(combined, json.load(fh))
    merged = write_best_hparams(real / "hpo" / "best_hparams.json", combined, fresh=False)
    n_models = sum(len(m) for v in merged.values() for m in v.values())
    print(f"  best_hparams.json: gemerged ({n_models} Modell-Eintraege gesamt)")

    # 2) hpo_log.csv: alle Temp-Logs konkatenieren, Key-Merge (variant,wmz,model).
    logs = [pd.read_csv(t / "hpo" / "hpo_log.csv")
            for t in temp_dirs if (t / "hpo" / "hpo_log.csv").is_file()]
    if logs:
        new = pd.concat(logs, ignore_index=True, sort=False)
        write_rows_csv(real / "hpo" / "hpo_log.csv", new,
                       ["variant", "wmz", "model"], fresh=False)
        print(f"  hpo_log.csv: {len(new)} neue Zeilen gemerged")

    # 3) stage10_metrics.csv: konkatenieren, Key-Merge (variant,wmz,model,stratum).
    mets = [pd.read_csv(t / "reports" / "stage10_metrics.csv")
            for t in temp_dirs if (t / "reports" / "stage10_metrics.csv").is_file()]
    if mets:
        new = pd.concat(mets, ignore_index=True, sort=False)
        write_rows_csv(real / "reports" / "stage10_metrics.csv", new,
                       ["variant", "wmz", "model", "stratum"], fresh=False)
        print(f"  stage10_metrics.csv: {len(new)} neue Zeilen gemerged")

    # 4) Qualitative Top-Listen je Job rueberkopieren (eine Datei pro Job).
    cnt = 0
    for t in temp_dirs:
        rep = t / "reports"
        if rep.is_dir():
            for q in rep.glob("stage10_qualitative_*.csv"):
                shutil.copy2(q, real / "reports" / q.name)
                cnt += 1
    print(f"  qualitative: {cnt} Dateien kopiert")
    print("Merge fertig.")


if __name__ == "__main__":
    main()
