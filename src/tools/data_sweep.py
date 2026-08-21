"""Daten-Sweep — Performance bei variierender Trainingsdatenmenge.

Trainiert *einen* Detektor (Klassik oder LSTM-AE) mit identischen
Hyperparametern auf unterschiedlich grossen Subsets des Trainings-
Tails und bewertet jeweils auf demselben (injizierten) Test-Set.

**Methodische Grundlage.** Stage 5 fixiert Validierung und Test
zeitlich (kein Shuffling). Variieren wir nur die Trainingsmenge,
bleiben Val/Test bit-identisch ueber alle Stufen — die Performance-
Differenzen sind dadurch sauber der Trainingsdatenmenge zuzuordnen
(Learning-Curve-Diagnose; Cortes u. a. 1994, Perlich u. a. 2003,
Viering und Loog 2023 - siehe methodology.md Sektion 7.8 und 9).

**Hyperparameter-Quelle.** Per Default werden die HPs aus
``outputs/<ds>/hpo/best_hparams.json`` fuer (variant, wmz, model)
geladen — das sind dieselben HPs, mit denen Stage 10 evaluiert. So
ist die Sweep-Stufe ``train_rows=0`` (volle Train-Menge) direkt mit
der Stage-10-Zeile dieser Konfiguration vergleichbar. Mit
``--hp key=value`` lassen sich einzelne HPs ueberschreiben (fuer
Ablationen / Smoke-Tests).

**Train-Subset-Konvention.** Wir nehmen den **juengsten zusammen-
haengenden Tail** der Flag-bereinigten Train-Indizes (kein Zufalls-
subsampling), damit:
1. Der LSTM-AE seine gleitenden Fenster behaelt — Zufallsstichproben
   wuerden die Sequenz-Struktur zerstoeren.
2. Der Subset zeitlich am naechsten an Val/Test liegt, was die
   realistische "wie wenig Daten reichen?"-Frage beantwortet (im
   Gegensatz zu "irgendein zufaelliges historisches Stueck").
3. Klassiker und LSTM-AE auf exakt denselben Stunden trainieren,
   was den modelluebergreifenden Vergleich fair haelt.

Beispiele:

    # LSTM-AE Sweep auf trend/wmz_1 mit besten HPs aus Stage 8:
    python src\\data_sweep.py --variant trend --wmz wmz_1 --model lstm_ae \\
        --train-rows 2000 5000 10000 0

    # IsolationForest Sweep auf raw/wmz_1, HPs aus best_hparams.json:
    python src\\data_sweep.py --variant raw --wmz wmz_1 --model iforest \\
        --train-rows 2000 5000 10000 0

    # Explizite HPs ueberschreiben (z. B. fuer schnelle Ablation):
    python src\\data_sweep.py --variant trend --wmz wmz_1 --model lstm_ae \\
        --train-rows 5000 --hp window_size=24 hidden_size=16 epochs=20

    # Smoke (kleine Stufe, kleine HPs):
    python src\\data_sweep.py --smoke --variant raw --wmz wmz_1 --model lstm_ae

Output:
    outputs/<ds>/reports/data_sweep.csv   (append-only, eine Zeile pro Stufe)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Dieses Tool liegt in src/tools/, importiert aber Sibling-Module aus src/
# (evaluation, models, stage7_train). Daher src/ auf den Pfad legen, bevor
# diese importiert werden.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation import evaluate, threshold_from_quantile
from models.registry import REGISTRY, WMZ_NAMES, model_features
from stage7_train import select_training_rows

ROOT = Path(__file__).resolve().parents[2]   # src/tools/ -> Projekt-Wurzel
DEFAULT_DATASET = "gebaeude_a"

# Name -> Klasse aus der REGISTRY abgeleitet (zentrale Wahrheit). Eine
# Klasse kann in mehreren Varianten registriert sein - hier reicht die
# erste Fundstelle, weil das Mapping nur den Namen aufloest.
NAME_TO_CLASS: dict[str, type] = {}
for _classes in REGISTRY.values():
    for _cls in _classes:
        NAME_TO_CLASS.setdefault(_cls.name, _cls)


def parse_kv(items: list[str]) -> dict:
    """Argparse-Helper: ``key=value`` -> dict mit (best-effort) Typkonversion.

    Macht ``window_size=48 epochs=30`` zu ``{"window_size": 48,
    "epochs": 30}``. Versucht erst int, dann float, dann String.
    """
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"  --hp erwartet 'key=value', bekommen: {item!r}")
        k, v = item.split("=", 1)
        # Typ erraten - int vor float, sonst String. Bool wird selten
        # gebraucht und ist hier bewusst nicht abgedeckt.
        for caster in (int, float):
            try:
                out[k] = caster(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--variant", default="raw", choices=list(REGISTRY))
    p.add_argument("--wmz", default="wmz_1", choices=WMZ_NAMES)
    p.add_argument("--model", default="lstm_ae",
                   choices=sorted(NAME_TO_CLASS.keys()),
                   help="Detektor-Name (zscore/lof/iforest/pelt/lstm_ae).")
    p.add_argument("--train-rows", type=int, nargs="+",
                   default=[2000, 5000, 10000, 0],
                   help="Trainings-Stundenzahl pro Stufe. 0 = alle "
                        "verfuegbaren (Flag-bereinigten) Train-Stunden.")
    p.add_argument("--threshold-quantile", type=float, default=0.99,
                   help="Schwellwert = q-Quantil der Validation-Scores "
                        "(score-basierte Modelle). PELT ist binaer (0.5).")
    p.add_argument("--hp", nargs="*", default=[],
                   help="HP-Overrides als key=value (z. B. window_size=48 "
                        "epochs=30). Schreibt ueber die HPs aus "
                        "best_hparams.json.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-Modus: eine kleine Stufe (3 000 Zeilen), "
                        "minimale HPs - prueft Pipeline-Lauf in <2 min.")
    p.add_argument("--no-load-best", action="store_true",
                   help="best_hparams.json ignorieren, nur --hp / Modell-"
                        "Defaults verwenden.")
    return p.parse_args()


def gt_kind_col(variant: str, wmz: str) -> str:
    """Schiene fuer die Eligibility-Ground-Truth: raw/residual gegen
    stationaere, trend gegen nicht-stationaere Anomalien (Methodik 6.3)."""
    return f"gt_{'nonstat' if variant == 'trend' else 'stat'}_{wmz}"


def load_best_hp(out_root: Path, variant: str, wmz: str,
                 model_name: str) -> dict:
    """HPs aus best_hparams.json laden; leeres dict, wenn nichts gespeichert."""
    path = out_root / "hpo" / "best_hparams.json"
    if not path.is_file():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return ((data.get(variant, {}) or {}).get(wmz, {}) or {}) \
        .get(model_name, {}).get("hparams", {}) or {}


def main() -> None:
    """Einen Detektor auf verschieden grossen Trainingsmengen bewerten.

    Die Hyperparameter bleiben ueber alle Stufen gleich (aus
    ``best_hparams.json``), variiert wird nur die Zahl der Trainingsstunden.
    Bewertet wird jedes Mal auf demselben injizierten Test-Set, damit die
    Unterschiede allein auf die Datenmenge zurueckgehen. Die Ergebnisse
    werden kumulativ in ``reports/data_sweep.csv`` gemergt.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # HP-Auswahl: best_hparams.json als Basis (vergleichbar mit Stage 10),
    # CLI-Overrides obendrauf. ``--no-load-best`` schaltet die Basis ab,
    # was nuetzlich ist, wenn man absichtlich Modell-Defaults testen will.
    if args.no_load_best:
        hp_base: dict = {}
    else:
        hp_base = load_best_hp(out_root, args.variant, args.wmz, args.model)
    hp = {**hp_base, **parse_kv(args.hp)}

    # Smoke-Override: minimal-invasiv, klein und schnell. Faengt vor allem
    # die Pipeline-Verifikation auf einer neuen Maschine ab.
    if args.smoke:
        args.train_rows = [3000]
        if args.model == "lstm_ae":
            hp = {"window_size": 24, "hidden_size": 16,
                  "n_layers": 1, "epochs": 20, **hp}

    ModelClass = NAME_TO_CLASS[args.model]
    is_pelt = (args.model == "pelt")

    # Variante C ist univariat (Trend-only) - PELT ist hier Default, LSTM-AE
    # ebenfalls erlaubt; die anderen Klassiker passen konzeptionell nicht
    # (sie wuerden auf einem einzigen Feature laufen). Wir erlauben es
    # trotzdem, schreiben aber eine Warnung, damit Sweeps nicht unbemerkt
    # unsinnige Konfigurationen produzieren.
    if args.variant == "trend" and args.model not in {"pelt", "lstm_ae"}:
        print(f"  Warnung: {args.model} auf Variante 'trend' ist konzep-"
              f"tionell unueblich (univariat). Lauf laeuft trotzdem.")

    print(f"Dataset:  {args.dataset}")
    print(f"Job:      {args.variant}/{args.wmz}/{args.model}")
    print(f"Stufen:   {args.train_rows} (0 = volle Train-Menge)")
    print(f"HPs:      {hp if hp else '(Modell-Defaults)'}")
    if hp_base and not args.no_load_best:
        print(f"  Basis aus best_hparams.json, Overrides aus --hp.")

    # --- Daten laden -------------------------------------------------------
    split = pd.read_parquet(parquet_dir / "split_assignment.parquet")["split"]
    clean = pd.read_parquet(parquet_dir / f"stage6_normalized_{args.variant}.parquet")
    inj = pd.read_parquet(parquet_dir / f"stage9_injected_{args.variant}.parquet")
    gt = pd.read_parquet(parquet_dir / "stage9_ground_truth.parquet")

    val_index = split.index[split == "val"]
    test_index = split.index[split == "test"]

    X_clean = model_features(clean, args.variant, args.wmz)
    # Volle Flag-bereinigte Train-Indizes als Obergrenze. Die Tail-
    # Selektion fuer jede Stufe sind die letzten N Eintraege davon.
    full_train_idx = select_training_rows(clean, args.wmz, args.variant,
                                          max_rows=None)
    full_n = len(full_train_idx)
    print(f"\nVolle Train-Stunden (Flag-bereinigt): {full_n}")

    # Eligibilitaet im Test-Set (ohne no_data, ohne known sensor issue) -
    # exakt dieselbe Maske wie Stage 10.
    elig = (~gt.loc[test_index, f"gt_no_data_{args.wmz}"].astype(bool)
            & ~gt.loc[test_index, f"gt_known_sensor_issue_{args.wmz}"].astype(bool))
    elig_idx = test_index[elig.to_numpy()]
    y_true = gt.loc[elig_idx, gt_kind_col(args.variant, args.wmz)].notna()

    rows: list[dict] = []

    print("\n" + "=" * 70)
    print("Daten-Sweep")
    print("=" * 70)

    for n in args.train_rows:
        n_eff = full_n if n == 0 else min(n, full_n)
        idx = full_train_idx[-n_eff:]   # juengster zusammenhaengender Tail
        X_train = X_clean.loc[idx]

        print(f"\n  Stufe: train_rows={n_eff} "
              f"(gewuenscht {n}, max {full_n})")
        t0 = time.time()
        model = ModelClass(**hp).fit(X_train)
        fit_secs = time.time() - t0
        # Device-Tag nur fuer Modelle, die ein Backend halten (LSTM-AE) -
        # bei Klassikern bleibt der Tag leer, damit der Log nicht "[None]"
        # haengt.
        device = getattr(model, "_device", None) or "cpu"
        print(f"    fit   : {fit_secs:6.1f} s  device={device}")

        # Schwellwert wie in Stage 10: PELT binaer (0.5), sonst q-Quantil
        # auf sauberen Val-Scores. Damit ist die Sweep-Metrik direkt mit
        # der Stage-10-Tabelle vergleichbar.
        t0 = time.time()
        if is_pelt:
            threshold = 0.5
        else:
            val_scores = model.score(X_clean.loc[val_index])
            threshold = threshold_from_quantile(val_scores, args.threshold_quantile)
        # Auf injiziertem Test scoren; Auswertung nur auf eligiblen Stunden.
        test_scores = model.score(inj).reindex(elig_idx)
        metrics = evaluate(test_scores, y_true, threshold, adjust=True)
        eval_secs = time.time() - t0
        print(f"    score : {eval_secs:6.1f} s  "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} AUC={metrics['roc_auc']:.3f}")

        rows.append({
            "variant": args.variant,
            "wmz": args.wmz,
            "model": args.model,
            "train_rows": n_eff,
            "fraction_of_full": round(n_eff / full_n, 4) if full_n else float("nan"),
            "fit_seconds": round(fit_secs, 2),
            "score_seconds": round(eval_secs, 2),
            "device": device,
            # HPs als ``hp_<key>``-Spalten, damit man pro Stufe die
            # tatsaechlich verwendeten Werte rekonstruieren kann (gerade
            # wenn HP-Overrides oder best_hparams-Werte ueberschrieben
            # haben).
            **{f"hp_{k}": v for k, v in hp.items()},
            **metrics,
        })

    df = pd.DataFrame(rows)
    out_path = reports_dir / "data_sweep.csv"
    if out_path.is_file():
        # An bestehende Datei anhaengen, damit Sweeps fuer verschiedene
        # (Variant, WMZ, Modell)-Kombinationen kumulativ gesammelt
        # werden - das CSV ist dann der zentrale Ort, an dem die
        # Learning-Curve-Daten landen.
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True, sort=False)
    df.to_csv(out_path, index=False)
    print(f"\n  Wrote (append) {out_path}  -> {len(df)} Zeilen gesamt")


if __name__ == "__main__":
    main()
