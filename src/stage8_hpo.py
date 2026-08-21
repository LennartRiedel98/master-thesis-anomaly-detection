"""Stage 8 - Hyperparameter-Optimierung (Grid-Search auf der Validierung).

Da die Modelle unueberwacht sind, gibt es auf realen Daten kein
Zielsignal. Loesung (Methodik 7.5/7.6, mit Betreuer abgestimmt): Wir
injizieren synthetische Anomalien in die **Validation**-Menge - mit einem
eigenen Seed, der sich vom Test-Seed (Stage 9) unterscheidet, damit die
HPs nicht auf die exakten Test-Anomalien getunt werden. Das Test-Set wird
hier nie angefasst.

Vorgehen pro (Variante x WMZ x Modell):
    1. Modell mit jeder HP-Kombination des Grids auf den sauberen
       Train-Zeilen (Flag-gefiltert) fitten.
    2. Auf der injizierten Validierung scoren.
    3. Score-basierte Modelle (zscore/lof/iforest/lstm_ae): threshold-freie
       ROC-AUC gegen die Ground-Truth. PELT (binaer): point-adjusted F1
       gegen die nicht-stationären Events.
    4. Beste Kombination je (Variante, WMZ, Modell) merken.

Ground-Truth-Schiene: Varianten A/B werden gegen die **stationären**
Anomalien bewertet, Variante C gegen die **nicht-stationären** - exakt die
Aufteilung aus Methodik 6.3.

Die Ausgaben werden **nicht-destruktiv** geschrieben: ein Teillauf (z. B.
``--models lstm_ae``) merged seine Ergebnisse in bestehende Dateien, ohne
die Eintraege anderer Modelle zu verlieren. ``--fresh`` erzwingt einen
sauberen Neuschrieb.

Output:
    outputs/<ds>/hpo/best_hparams.json
    outputs/<ds>/hpo/hpo_log.csv   (alle Kombinationen mit Score)
"""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from injection_apply import build_injected
from evaluation import roc_auc, pr_auc, evaluate, threshold_from_quantile
from models.registry import REGISTRY, WMZ_NAMES, model_features, iter_training_jobs
from result_io import write_best_hparams, write_rows_csv
from stage7_train import select_training_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "gebaeude_a"

# Eigener Seed fuer die Validierungs-Injektion (!= Test-Seed in Stage 9).
VAL_SEED = 8

# Seed fuer das Random-Search-Sampling (reproduzierbar; identische
# Kandidaten-Menge ueber alle LSTM-Jobs -> vergleichbar).
SEARCH_SEED = 17
LSTM_SEARCH_ITER_DEFAULT = 25

# --- Suchstrategie je Modell ----------------------------------------------
#
# Niedrigdimensionale, billige Modelle -> erschoepfendes GRID (HP_GRIDS).
# Der hochdimensionale, teure LSTM-AE -> RANDOM SEARCH ueber HP_SPACES
# (Bergstra und Bengio 2012: bei gleichem Budget effizienter als Grid,
# sobald nicht alle Dimensionen gleich wichtig sind).
#
# Bewusst NICHT getunt: ``contamination`` (LOF/IsolationForest). Es
# beeinflusst nur das interne predict()-Schwellwert-Attribut, nicht das
# Ranking von score_samples(). Da die HPO rangbasierte ROC-AUC nutzt und
# die finale Schwelle in Stage 10 datengetrieben (Val-Quantil) gesetzt
# wird, hat contamination keinen Effekt auf unsere Ergebnisse.
HP_GRIDS: dict[str, list[dict]] = {
    "zscore":  [{"aggregation": a} for a in ("max", "l2", "mean")],
    "lof":     [{"n_neighbors": k} for k in (5, 10, 20, 40, 80)],
    "iforest": [{"n_estimators": n, "max_features": mf}
                for n in (100, 200, 400) for mf in (0.5, 0.7, 1.0)],
    "pelt":    [{"penalty": p} for p in (5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0)],
    # Konstanz-/Plateau-Detektor: kurzes Konstanz-Fenster x langes Kontext-
    # fenster x Einbruchs-Empfindlichkeit. Reines pandas (Rolling-Std),
    # CPU-only - der 3x2x2-Grid ist in Sekunden durch.
    "constancy": [{"window": w, "baseline_window": bw, "sensitivity": s}
                  for w in (4, 6, 12) for bw in (72, 168) for s in (0.5, 1.0)],
}

# Random-Search-Raum des LSTM-AE (volles Kreuzprodukt waere 3*3*2*4*3 = 216
# Konfigurationen je Job; per Random Search ziehen wir LSTM_SEARCH_ITER
# davon).
HP_SPACES: dict[str, dict[str, list]] = {
    "lstm_ae": {
        "window_size": [24, 48, 72],
        "hidden_size": [16, 32, 64],
        "n_layers": [1, 2],
        "learning_rate": [1e-2, 3e-3, 1e-3, 3e-4],
        "epochs": [30, 50, 100],
    },
}

# --- HPO-Option A: gezielter Grid-Search nur ueber window x hidden ---------
#
# Motivation: Der Daten-Sweep mit window=48 erreichte auf
# trend/wmz_1 F1=0,776, waehrend das Random-Search-HPO (25 von 216 Configs)
# window=24 waehlte (F1=0,611). Frage: War window die unter-abgetastete
# Dimension oder war Random Search fair? Dieser erschoepfende 3x3-Grid ueber
# window_size x hidden_size beantwortet das direkt; die uebrigen HPs werden
# auf empirisch gute Defaults aus hpo_log.csv fixiert, damit nur die zwei
# fraglichen Achsen variieren (9 Configs/Job statt 216).
LSTM_GRIDWH_WINDOW = [24, 48, 72]
LSTM_GRIDWH_HIDDEN = [16, 32, 64]
LSTM_GRIDWH_FIXED = {"n_layers": 2, "learning_rate": 1e-3, "epochs": 50}


def lstm_gridwh_configs() -> list[dict]:
    """3 x 3 = 9 Konfigurationen (window_size x hidden_size), restliche HPs fix."""
    return [{"window_size": w, "hidden_size": h, **LSTM_GRIDWH_FIXED}
            for w in LSTM_GRIDWH_WINDOW for h in LSTM_GRIDWH_HIDDEN]


def sample_space(space: dict[str, list], rng: np.random.Generator,
                 n_iter: int) -> list[dict]:
    """Zieht bis zu ``n_iter`` *eindeutige* Konfigurationen aus ``space``.

    Begrenzt automatisch auf die Gesamt-Gittergroesse (kein Sinn, mehr zu
    ziehen als es Kombinationen gibt) und vermeidet Duplikate, damit kein
    GPU-Budget auf doppelte Trainings verschwendet wird.
    """
    keys = list(space)
    total = math.prod(len(space[k]) for k in keys)
    n = min(n_iter, total)
    seen: set[tuple] = set()
    out: list[dict] = []
    while len(out) < n:
        cfg = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        signature = tuple(cfg[k] for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(cfg)
    return out


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen.

    Wichtig sind hier ``--hpo-seeds`` (mehrere Startwerte je
    Hyperparameter-Kombination, um die Seed-Streuung des LSTM-AE sichtbar zu
    machen) und ``--device`` fuer die GPU-Laeufe.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--variants", nargs="+", default=None, choices=list(REGISTRY))
    p.add_argument("--wmz", nargs="+", default=None, choices=WMZ_NAMES)
    p.add_argument("--models", nargs="+", default=None,
                   help="Nur diese Modellnamen (z. B. zscore lof iforest pelt). "
                        "Default: alle. Nuetzlich, um z. B. lstm_ae auf "
                        "Maschinen ohne GPU auszulassen.")
    p.add_argument("--max-train-rows", type=int, default=None, metavar="N",
                   help="Trainings-Stunden begrenzen (Probelauf).")
    p.add_argument("--lstm-search-iter", type=int, default=LSTM_SEARCH_ITER_DEFAULT,
                   metavar="N",
                   help=f"Anzahl Random-Search-Konfigurationen fuer den "
                        f"LSTM-AE (nur bei --lstm-strategy random). "
                        f"Default: {LSTM_SEARCH_ITER_DEFAULT}.")
    p.add_argument("--lstm-strategy", choices=["random", "gridwh"],
                   default="random",
                   help="Suchstrategie fuer den LSTM-AE. 'random' (Default): "
                        "Random Search ueber HP_SPACES mit --lstm-search-iter "
                        "Ziehungen. 'gridwh': erschoepfender 3x3-Grid "
                        "window_size x hidden_size (n_layers=2, lr=1e-3, "
                        "epochs=50 fix) = HPO-Option A, 9 Configs/Job.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Device fuer den LSTM-AE. Default 'auto' (CUDA>MPS>CPU). "
                        "Auf dem M4-Pro-Mac 'cpu' erzwingen - der nn.LSTM-Kernel "
                        "auf MPS ist defekt (OOM > 60 GB).")
    p.add_argument("--hpo-metric", default="roc_auc",
                   choices=["roc_auc", "pr_auc", "f1"],
                   help="Auswahl-Kriterium fuer score-basierte Modelle. "
                        "'roc_auc' (Default), 'pr_auc' (trennschaerfer bei "
                        "seltenen Anomalien) oder 'f1' (point-adjusted, an der "
                        "Val-Quantil-Schwelle; richtet die HPO auf die Testmetrik "
                        "aus). PELT nutzt unabhaengig davon F1.")
    p.add_argument("--hpo-seeds", type=int, default=1, metavar="N",
                   help="Seed-Averaging: jede Config N-mal mit verschiedenen "
                        "random_state fitten und die Val-Metrik mitteln "
                        "(nur Modelle mit random_state, z. B. LSTM-AE). "
                        "Default 1 (kein Averaging).")
    p.add_argument("--threshold-quantile", type=float, default=0.99, metavar="Q",
                   help="Quantil fuer die F1-Schwelle bei --hpo-metric f1 "
                        "(Default 0.99, wie Stage 10).")
    p.add_argument("--fresh", action="store_true",
                   help="best_hparams.json/hpo_log.csv komplett neu schreiben "
                        "statt in Bestehendes zu mergen (sauberer Rebuild).")
    return p.parse_args()


def gt_column(variant: str, wmz: str) -> str:
    """Welche Ground-Truth-Spalte gilt fuer diese Variante?

    A/B (raw/residual) -> stationäre Anomalien; C (trend) -> nicht-stationär.
    """
    schiene = "nonstat" if variant == "trend" else "stat"
    return f"gt_{schiene}_{wmz}"


def _free_accelerator_cache() -> None:
    """Gibt belegten GPU-Cache frei - zwischen einzelnen Fits aufzurufen.

    Kritisch auf Apple-Silicon/MPS: Der MPS-Allokator behaelt den Speicher
    abgeschlossener Fits im Cache und gibt ihn - anders als CUDA unter
    VRAM-Druck - **nicht** automatisch zurueck. Bei den vielen sequentiellen
    LSTM-Fits eines HPO-Laufs (bis zu 9 Configs x 9 Jobs) akkumuliert das
    sonst zu einem Footprint von zig GB (auf einem 48-GB-Mac in den Swap).
    ``torch`` wird nur angefasst, wenn es ueberhaupt schon importiert wurde
    (die klassischen Modelle brauchen es nicht).
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        # Cache-Freigabe ist rein optional - niemals den Lauf daran scheitern lassen.
        pass


def score_combo(model, X_val: pd.DataFrame, y_true: pd.Series,
                is_changepoint: bool, metric: str = "roc_auc",
                f1_quantile: float = 0.99) -> float:
    """Eine HP-Kombination auf der Validierung bewerten. Hoeher = besser.

    ``metric`` steuert das Auswahl-Kriterium fuer score-basierte Modelle:
      * ``roc_auc`` (Default): threshold-freie ROC-AUC.
      * ``pr_auc``: Average Precision - bei seltenen Anomalien trennschaerfer,
        weil sie die seltene Positiv-Klasse fokussiert statt beide zu mitteln
        (Davis und Goadrich 2006; Saito und Rehmsmeier 2015).
      * ``f1``: point-adjusted F1 an der Val-Quantil-Schwelle - richtet die
        HPO direkt auf die Testmetrik aus (F1-Maximierung via Schwelle:
        Lipton u. a. 2014; ein ROC-AUC-Optimum maximiert F1 nicht
        zwangslaeufig: Davis und Goadrich 2006).
    PELT (Change-Point, binaer) wird davon unabhaengig ueber point-adjusted
    F1 bewertet.
    """
    scores = model.score(X_val)
    if is_changepoint:
        # PELT liefert binaere Change-Points -> point-adjusted F1.
        return evaluate(scores, y_true, threshold=0.5, adjust=True)["f1"]
    yt = y_true.to_numpy(dtype=bool)
    if metric == "roc_auc":
        return roc_auc(scores, yt)
    if metric == "pr_auc":
        return pr_auc(scores, yt)
    if metric == "f1":
        thr = threshold_from_quantile(scores, f1_quantile)
        return evaluate(scores, y_true, threshold=thr, adjust=True)["f1"]
    raise ValueError(f"unbekannte --hpo-metric: {metric}")


def fit_and_score(model_cls, hp: dict, X_train, X_val, y_true, is_cp: bool,
                  metric: str, n_seeds: int, f1_quantile: float
                  ) -> tuple[float, list[tuple[int | None, float]]]:
    """Fittet eine Config und gibt ``(mean, per_seed)`` zurueck.

    Seed-Averaging: stochastische Modelle (zufaellige Initialisierung +
    Batch-Reihenfolge, v. a. der LSTM-AE) liefern je nach ``random_state``
    schwankende Val-Werte; ein einzelner Seed ist eine verrauschte
    Schaetzung, die die HP-Rangfolge kippen kann (Reimers und Gurevych 2017;
    Bouthillier u. a. 2021). Bei ``n_seeds > 1`` wird die Config mit mehreren
    Seeds gefittet und die Metrik gemittelt - aber nur fuer Modelle, die
    ueberhaupt ein ``random_state`` akzeptieren; deterministische Modelle
    (LOF, Z-Score, PELT) werden genau einmal gefittet.

    Rueckgabe ist ein Tupel ``(mean, per_seed)``: ``mean`` ist die
    Selektionsmetrik (so wie bisher als float fuer den HP-Vergleich),
    ``per_seed`` ist die Liste der einzelnen Seed-Scores als
    ``(seed_id, score)``-Paare. Damit kann der Aufrufer in ``hpo_log.csv``
    **eine Zeile pro Seed** schreiben statt nur den Mittelwert - die
    Inter-Seed-Streuung wird so im Log persistiert und steht der
    Seed-Varianz-Analyse (``src/tools/seed_variance.py``) als echte
    Std-/Min-/Max-Aussage zur Verfuegung. ``seed_id`` ist ``None`` fuer
    deterministische Modelle (genau ein Fit, keine Seed-Achse).

    Die Speicherfreigabe (GPU-Cache/gc) erfolgt nach **jedem** Fit, damit der
    Footprint ueber die vielen Fits nicht akkumuliert.
    """
    accepts_seed = "random_state" in inspect.signature(model_cls).parameters
    seeds = list(range(n_seeds)) if (accepts_seed and n_seeds > 1) else [None]
    per_seed: list[tuple[int | None, float]] = []
    for i in seeds:
        h = dict(hp)
        if i is not None:
            h["random_state"] = 1000 + i   # distinkte, reproduzierbare Seeds
        model = None
        try:
            model = model_cls(**h).fit(X_train)
            v = score_combo(model, X_val, y_true, is_cp, metric, f1_quantile)
        finally:
            del model
            gc.collect()
            _free_accelerator_cache()
        per_seed.append((1000 + i if i is not None else None, float(v)))
    clean = [v for _, v in per_seed if v == v]   # NaN entfernen
    mean = float(np.mean(clean)) if clean else float("nan")
    return mean, per_seed


def main() -> None:
    """Stage 8 ausfuehren: Gittersuche auf der injizierten Validierungsmenge.

    Da die Detektoren unueberwacht arbeiten, gibt es auf realen Daten kein
    Zielsignal, gegen das sich Hyperparameter optimieren liessen. Deshalb
    werden synthetische Anomalien in die *Validierung* injiziert - mit einem
    anderen Startwert als in Stage 9, damit die Wahl nicht auf die exakten
    Test-Anomalien passt. Das Test-Set wird hier nie beruehrt.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    scaler_dir = out_root / "scalers"
    hpo_dir = out_root / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_parquet(parquet_dir / "split_assignment.parquet")["split"]

    print(f"Dataset: {args.dataset}")
    print(f"  Injiziere Anomalien in die Validierung (Seed {VAL_SEED}) ...")
    inj_frames, gt = build_injected(parquet_dir, scaler_dir, split, "val", VAL_SEED)
    val_index = split.index[split == "val"]
    print(f"  Validierungs-Stunden: {len(val_index)}")

    sel_variants = set(args.variants) if args.variants else set(REGISTRY)
    sel_wmz = set(args.wmz) if args.wmz else set(WMZ_NAMES)
    sel_models = set(args.models) if args.models else None

    # LSTM-Kandidaten einmalig festlegen (gleiche Menge fuer alle Jobs eines
    # Modells -> reproduzierbar und vergleichbar). Strategie waehlbar:
    #   random  -> Random Search ueber HP_SPACES (Default)
    #   gridwh  -> erschoepfender 3x3-Grid window x hidden (HPO-Option A)
    if args.lstm_strategy == "gridwh":
        sampled = {"lstm_ae": lstm_gridwh_configs()}
        print(f"  LSTM-Strategie: gridwh (9 Configs/Job, window x hidden)")
    else:
        search_rng = np.random.default_rng(SEARCH_SEED)
        sampled = {name: sample_space(space, search_rng, args.lstm_search_iter)
                   for name, space in HP_SPACES.items()}
        print(f"  LSTM-Strategie: random ({args.lstm_search_iter} Configs/Job)")

    # Device fuer den LSTM-AE optional erzwingen (z. B. 'cpu' auf dem Mac).
    if args.device != "auto" and "lstm_ae" in sampled:
        sampled["lstm_ae"] = [{**c, "device": args.device}
                              for c in sampled["lstm_ae"]]
        print(f"  LSTM-Device erzwungen: {args.device}")

    def configs_for(model_name: str) -> list[dict]:
        """Grid (klassisch) oder gezogene Random-Search-Kandidaten (LSTM)."""
        if model_name in HP_GRIDS:
            return HP_GRIDS[model_name]
        return sampled.get(model_name, [{}])

    # Saubere Train-Frames je Variante cachen.
    clean_cache: dict[str, pd.DataFrame] = {}

    def clean_frame(variant: str) -> pd.DataFrame:
        """Unveraenderte (nicht injizierte) Merkmalstabelle einer Schiene liefern.

        Wird fuer den Fit gebraucht: Trainiert wird auf sauberen Daten,
        bewertet auf der injizierten Validierung.
        """
        if variant not in clean_cache:
            clean_cache[variant] = pd.read_parquet(
                parquet_dir / f"stage6_normalized_{variant}.parquet")
        return clean_cache[variant]

    best: dict = {}
    log_rows: list[dict] = []

    print(f"  Auswahl-Kriterium: {args.hpo_metric}"
          + (f" (Schwelle q={args.threshold_quantile})" if args.hpo_metric == "f1" else "")
          + f" | Seed-Averaging: {args.hpo_seeds}x")

    print("\n" + "=" * 70)
    print("Stage 8 - HPO")
    print("=" * 70)

    for variant, wmz, model_cls in iter_training_jobs():
        model_name = model_cls.name
        if (variant not in sel_variants or wmz not in sel_wmz
                or (sel_models and model_name not in sel_models)):
            continue

        clean = clean_frame(variant)
        X_train_full = model_features(clean, variant, wmz)
        train_idx = select_training_rows(clean, wmz, variant, args.max_train_rows)
        X_train = X_train_full.loc[train_idx]

        # Injizierte Validierung fuer diese Variante (region-Slice).
        X_val = inj_frames[variant].loc[val_index]
        gt_kind = gt.loc[val_index, gt_column(variant, wmz)]
        if model_name == "constancy":
            # Typ-spezifischer Detektor: gegen die EIGENE Ziel-Anomalie
            # (Plateau) tunen, nicht gegen die gesamte stationäre Schiene -
            # sonst optimiert die roc_auc-Selektion auf inzidentelle Treffer
            # bei drop/leakage und waehlt plateau-schlechtere HPs. Analog
            # zur PELT-Sonderbehandlung (eigene Nische, eigene Selektion).
            # Bewertet wird in Stage 10 dennoch am eingefrorenen Gesamt-
            # Lineal (§ 7.7) - hier geht es nur um die HP-Wahl.
            y_true = (gt_kind == "plateau")
        else:
            y_true = gt_kind.notna()
        is_cp = (model_name == "pelt")

        grid = configs_for(model_name)
        best_score, best_hp = float("-inf"), grid[0]
        for hp in grid:
            try:
                s, per_seed = fit_and_score(
                    model_cls, hp, X_train, X_val, y_true, is_cp,
                    args.hpo_metric, args.hpo_seeds, args.threshold_quantile)
            except NotImplementedError:
                s, per_seed = float("nan"), []
            # Selektion: weiter ueber den Seed-Mittel (Methodik 7.5),
            # damit die HP-Wahl unabhaengig vom Logging-Format ist.
            if s == s and s > best_score:
                best_score, best_hp = s, hp
            # Logging: pro Seed eine Zeile in ``hpo_log.csv``. Bei
            # deterministischen Modellen (per_seed = [(None, score)])
            # bleibt es bei einer Zeile mit ``seed = NaN`` - identisch
            # zum bisherigen Schema. Bei ``--hpo-seeds > 1`` werden
            # mehrere Zeilen mit identischen HPs, aber unterschiedlichem
            # ``seed`` und ``score`` geschrieben - die Inter-Seed-Std
            # ist damit aus der CSV rekonstruierbar.
            log_entries = per_seed if per_seed else [(None, s)]
            for seed_id, val in log_entries:
                log_rows.append({"variant": variant, "wmz": wmz,
                                 "model": model_name,
                                 "seed": seed_id if seed_id is not None
                                         else float("nan"),
                                 "score": val, **hp})
            seeds_tag = (f"  (mean over {len(per_seed)} seeds)"
                         if len(per_seed) > 1 else "")
            print(f"  {variant}/{wmz}/{model_name:<8} {str(hp):<40} "
                  f"score={s:.4f}{seeds_tag}")

        best.setdefault(variant, {}).setdefault(wmz, {})[model_name] = {
            "hparams": best_hp,
            "score": None if best_score == float("-inf") else best_score,
        }

    # --- Persistenz (nicht-destruktiv: in Bestehendes mergen, ausser --fresh) -
    # So ueberschreibt ein Teillauf (z. B. --models lstm_ae) die Ergebnisse
    # der anderen Modelle aus einem frueheren Lauf nicht.
    best_path = hpo_dir / "best_hparams.json"
    write_best_hparams(best_path, best, fresh=args.fresh)
    log_path = hpo_dir / "hpo_log.csv"
    write_rows_csv(log_path, pd.DataFrame(log_rows),
                   ["variant", "wmz", "model"], fresh=args.fresh)

    print("\n" + "-" * 70)
    mode = "neu geschrieben (--fresh)" if args.fresh else "in Bestehendes gemerged"
    print(f"  Beste HPs -> {best_path}  [{mode}]")
    print(f"  Voller Log -> {log_path}")


if __name__ == "__main__":
    main()
