"""Stage 10 - Evaluation der Detektoren auf dem injizierten Test-Set.

Setzt Stage 7 (Modelle), Stage 8 (beste HPs) und Stage 9 (injiziertes
Test-Set + Ground-Truth) zusammen und berechnet die in Methodik 10
beschriebenen Auswertungen:

10a Quantitativ (synthetische Anomalien, Kategorie B):
    Pro (Variante x WMZ x Modell): Precision/Recall/F1 (point-adjusted)
    und ROC-AUC auf den eligiblen Test-Stunden, plus Recall je
    Anomalietyp und je Typ@Intensitaet. "Eligibel" = ohne no-data und
    ohne bekannte Sensor-Fehler (Kategorie A), damit der B-Wert nicht
    von Kategorie-A-Stoerungen verzerrt wird.

10b Qualitativ:
    Stratifizierte Stichprobe der staerksten Detektionen je Modell als
    CSV fuer die manuelle TP/FP/Borderline-Bewertung.

10c Labelfreie Realwelt-Validierung:
    Hat PELT (Variante C) auf dem vollen Trend einen Change-Point nahe der
    EnSikuMaV (2022-09-01) gefunden?

Schwellwert: q-Quantil der Scores auf der *sauberen* Validierung
(Default 0.99) - unabhaengig vom Test-Set. PELT ist binaer (Schwelle 0.5).

stage10_metrics.csv wird **nicht-destruktiv** geschrieben: ein Teillauf
(z. B. ``--models lstm_ae``) ersetzt nur die Zeilen der bewerteten
Modelle und behaelt die uebrigen. ``--fresh`` erzwingt einen Neuschrieb.

Outputs:
    outputs/<ds>/reports/stage10_metrics.csv
    outputs/<ds>/reports/stage10_qualitative_<variant>_<wmz>_<model>.csv
    outputs/<ds>/reports/stage10_changepoints_regulatory.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import evaluate, threshold_from_quantile, segments
from models.registry import REGISTRY, WMZ_NAMES, model_features, iter_training_jobs
from result_io import write_rows_csv
from stage7_train import select_training_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "demo_synthetic"

REGULATORY_TS = pd.Timestamp("2022-09-01")
REGULATORY_TOL_DAYS = 30


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--variants", nargs="+", default=None, choices=list(REGISTRY))
    p.add_argument("--wmz", nargs="+", default=None, choices=WMZ_NAMES)
    p.add_argument("--models", nargs="+", default=None,
                   help="Nur diese Modellnamen (z. B. lstm_ae auslassen).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Device fuer den LSTM-AE beim Re-Fit. Default 'auto' "
                        "(CUDA>MPS>CPU). Auf dem M4-Pro-Mac 'cpu' erzwingen - "
                        "der nn.LSTM-Kernel auf MPS ist defekt (OOM).")
    p.add_argument("--threshold-quantile", type=float, default=0.99)
    p.add_argument("--max-train-rows", type=int, default=None, metavar="N")
    p.add_argument("--qualitative-top", type=int, default=50,
                   help="Anzahl Top-Detektionen je Modell fuer den 10b-Export.")
    p.add_argument("--fresh", action="store_true",
                   help="stage10_metrics.csv komplett neu schreiben statt in "
                        "Bestehendes zu mergen (sauberer Rebuild).")
    return p.parse_args()


def gt_kind_col(variant: str, wmz: str) -> str:
    """Name der Ground-Truth-Spalte fuer die Anomalieart einer Schiene.

    Schiene C (trend) zielt auf nicht-stationaere Anomalien, A und B auf
    stationaere - die Bewertung greift deshalb je Schiene auf eine andere
    Ground-Truth zu.
    """
    return f"gt_{'nonstat' if variant == 'trend' else 'stat'}_{wmz}"


def gt_label_col(variant: str, wmz: str) -> str:
    """Name der Ground-Truth-Spalte mit dem Ereignis-Label einer Schiene.

    Das Label traegt Typ und Intensitaet (etwa ``spike@10``) und erlaubt
    damit die nach Anomalietyp aufgeschluesselte Auswertung.
    """
    return f"gt_{'nonstat' if variant == 'trend' else 'stat'}_label_{wmz}"


def recall_by_group(scores: pd.Series, threshold: float,
                    group: pd.Series) -> dict[str, dict]:
    """Point-adjusted Recall je Gruppenwert (Anomalietyp bzw. Typ@Intensität).

    Ein Event-Segment gilt als erkannt, wenn irgendein Punkt darin den
    Schwellwert ueberschreitet. group ist eine Serie mit Label oder NA.
    """
    s = scores.reindex(group.index).to_numpy(dtype=float)
    hit = np.where(np.isnan(s), False, s > threshold)
    out: dict[str, dict] = {}
    for value in sorted(g for g in group.dropna().unique()):
        mask = (group == value).to_numpy()
        segs = segments(mask)
        detected = sum(1 for a, b in segs if hit[a:b].any())
        out[value] = {"n_events": len(segs), "detected": detected,
                      "recall": detected / len(segs) if segs else float("nan")}
    return out


def load_best_hparams(hpo_dir: Path) -> dict:
    """Beste Hyperparameter aus der Stage-8-Ausgabe laden.

    Fehlt der Eintrag fuer einen Job, faellt der Detektor auf seine
    Vorgabewerte zurueck - das passiert, wenn Stage 8 fuer diese Kombination
    nicht gelaufen ist.
    """
    path = hpo_dir / "best_hparams.json"
    if not path.is_file():
        print(f"  (kein {path.name} - nutze Default-HPs)")
        return {}
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    """Stage 10 ausfuehren: Detektoren auf dem injizierten Test-Set bewerten.

    Setzt Stage 7 (Modelle), Stage 8 (beste Hyperparameter) und Stage 9
    (injiziertes Test-Set samt Ground-Truth) zusammen. Berechnet je Job
    Precision, Recall und F1 nach Point-Adjust sowie die schwellenfreien
    Masse ROC-AUC und PR-AUC, zusaetzlich nach Anomalietyp und -intensitaet
    aufgeschluesselt. Die Schwelle stammt aus dem 99-%-Quantil der Scores auf
    der sauberen Validierung, wird also nicht am Test-Set gewaehlt.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_parquet(parquet_dir / "split_assignment.parquet")["split"]
    gt = pd.read_parquet(parquet_dir / "stage9_ground_truth.parquet")
    best = load_best_hparams(out_root / "hpo")

    val_index = split.index[split == "val"]
    test_index = split.index[split == "test"]

    sel_variants = set(args.variants) if args.variants else set(REGISTRY)
    sel_wmz = set(args.wmz) if args.wmz else set(WMZ_NAMES)
    sel_models = set(args.models) if args.models else None

    clean_cache: dict[str, pd.DataFrame] = {}
    inj_cache: dict[str, pd.DataFrame] = {}

    def clean_frame(variant: str) -> pd.DataFrame:
        """Unveraenderte Merkmalstabelle einer Schiene liefern (beim ersten Zugriff geladen).

        Gebraucht fuer die Schwellenbestimmung auf der sauberen Validierung.
        """
        if variant not in clean_cache:
            clean_cache[variant] = pd.read_parquet(
                parquet_dir / f"stage6_normalized_{variant}.parquet")
        return clean_cache[variant]

    def inj_frame(variant: str) -> pd.DataFrame:
        """Injizierte Merkmalstabelle einer Schiene liefern (beim ersten Zugriff geladen).

        Gebraucht fuer die eigentliche Bewertung auf dem gestoerten Test-Fenster.
        """
        if variant not in inj_cache:
            inj_cache[variant] = pd.read_parquet(
                parquet_dir / f"stage9_injected_{variant}.parquet")
        return inj_cache[variant]

    metric_rows: list[dict] = []
    cp_rows: list[dict] = []

    print(f"Dataset: {args.dataset}")
    print("=" * 70)
    print("Stage 10 - Evaluation")
    print("=" * 70)

    for variant, wmz, model_cls in iter_training_jobs():
        model_name = model_cls.name
        if (variant not in sel_variants or wmz not in sel_wmz
                or (sel_models and model_name not in sel_models)):
            continue

        # HPs: beste aus Stage 8, sonst Modell-Defaults.
        hp = (best.get(variant, {}).get(wmz, {})
              .get(model_name, {}).get("hparams", {})) or {}
        # Device nur fuer den LSTM-AE erzwingen (klassische Modelle kennen
        # den Parameter nicht). Wichtig auf dem Mac: 'cpu' statt MPS.
        if args.device != "auto" and model_name == "lstm_ae":
            hp = {**hp, "device": args.device}

        clean = clean_frame(variant)
        X_clean = model_features(clean, variant, wmz)
        train_idx = select_training_rows(clean, wmz, variant, args.max_train_rows)

        try:
            model = model_cls(**hp).fit(X_clean.loc[train_idx])
        except NotImplementedError:
            print(f"  {variant}/{wmz}/{model_name}: [stub] uebersprungen")
            continue

        is_cp = (model_name == "pelt")

        # Schwellwert aus sauberen Validierungs-Scores (PELT: binaer -> 0.5).
        if is_cp:
            threshold = 0.5
        else:
            val_scores = model.score(X_clean.loc[val_index])
            threshold = threshold_from_quantile(val_scores, args.threshold_quantile)

        # Auf dem injizierten Test scoren (volle Reihe fuer LSTM-Kontext,
        # Auswertung nur auf Test-Stunden).
        inj = inj_frame(variant)
        test_scores = model.score(inj).reindex(test_index)

        # Eligibilitaet: keine no-data- und keine Kategorie-A-Stunden.
        elig = (~gt.loc[test_index, f"gt_no_data_{wmz}"].astype(bool)
                & ~gt.loc[test_index, f"gt_known_sensor_issue_{wmz}"].astype(bool))
        kind = gt.loc[test_index, gt_kind_col(variant, wmz)]
        label = gt.loc[test_index, gt_label_col(variant, wmz)]

        # --- 10a Gesamt-Metrik (positiv = irgendeine Anomalie der Schiene) --
        elig_idx = test_index[elig.to_numpy()]
        y_true = kind.loc[elig_idx].notna()
        # Point-Adjust gilt fuer alle Intervall-Anomalien (Methodik 10a) -
        # auch fuer PELT, das nur den Change-Point-Onset markiert, waehrend
        # die nicht-stationäre Anomalie das ganze Folge-Intervall umfasst.
        overall = evaluate(test_scores.loc[elig_idx], y_true, threshold,
                           adjust=True)
        metric_rows.append({"variant": variant, "wmz": wmz, "model": model_name,
                            "stratum": "overall", **overall})

        # --- 10a Recall je Typ und je Typ@Intensität ------------------------
        for stratum_series, tag in ((kind.loc[elig_idx], "type"),
                                    (label.loc[elig_idx], "type_intensity")):
            for value, rec in recall_by_group(test_scores, threshold,
                                              stratum_series).items():
                metric_rows.append({"variant": variant, "wmz": wmz,
                                    "model": model_name, "stratum": f"{tag}:{value}",
                                    "recall": rec["recall"],
                                    "n_true": rec["n_events"],
                                    "n_pred": rec["detected"],
                                    "threshold": threshold})

        # Device-Tag nur, wenn das Modell ein Backend hat (LSTM-AE).
        dev_tag = f" [{model._device}]" if getattr(model, "_device", None) else ""
        print(f"  {variant}/{wmz}/{model_name:<8}{dev_tag} "
              f"P={overall['precision']:.3f} R={overall['recall']:.3f} "
              f"F1={overall['f1']:.3f} AUC={overall['roc_auc']:.3f}")

        # --- 10b Qualitativer Export (Top-Detektionen) ----------------------
        q = pd.DataFrame({
            "score": test_scores,
            "kind_stat": gt.loc[test_index, f"gt_stat_{wmz}"],
            "kind_nonstat": gt.loc[test_index, f"gt_nonstat_{wmz}"],
            "known_sensor_issue": gt.loc[test_index, f"gt_known_sensor_issue_{wmz}"],
            "no_data": gt.loc[test_index, f"gt_no_data_{wmz}"],
        }).sort_values("score", ascending=False).head(args.qualitative_top)
        q.to_csv(reports_dir / f"stage10_qualitative_{variant}_{wmz}_{model_name}.csv")

        # --- 10c PELT vs. EnSikuMaV (auf der vollen Trend-Reihe) ------------
        if is_cp and variant == "trend":
            full = model.score(inj)        # binaere Change-Points, volle Reihe
            cps = full.index[full == 1.0]
            near = [str(ts) for ts in cps
                    if abs((ts - REGULATORY_TS).days) <= REGULATORY_TOL_DAYS]
            cp_rows.append({"wmz": wmz, "n_changepoints": int((full == 1.0).sum()),
                            "near_ensikumav": "; ".join(near) or "keiner"})

    # --- Persistenz (nicht-destruktiv: in Bestehendes mergen, ausser --fresh) -
    # Ein Teillauf (z. B. --models lstm_ae) ersetzt nur die Zeilen der in
    # diesem Lauf bewerteten (variant, wmz, model)-Kombinationen; die
    # Ergebnisse anderer Modelle aus frueheren Laeufen bleiben erhalten.
    metrics = pd.DataFrame(metric_rows)
    metrics_path = reports_dir / "stage10_metrics.csv"
    metrics = write_rows_csv(metrics_path, metrics,
                             ["variant", "wmz", "model"], fresh=args.fresh)
    mode = "neu" if args.fresh else "gemerged"
    print(f"\n  Metriken -> {metrics_path}  ({len(metrics)} Zeilen, {mode})")

    if cp_rows:
        cp_path = reports_dir / "stage10_changepoints_regulatory.csv"
        write_rows_csv(cp_path, pd.DataFrame(cp_rows), ["wmz"], fresh=args.fresh)
        print(f"  Change-Points (10c) -> {cp_path}")

    # Kompakte Konsolen-Zusammenfassung der Gesamt-Metriken.
    overall = metrics[metrics["stratum"] == "overall"]
    if not overall.empty:
        print("\n  Gesamt-F1 je Variante (Mittel ueber WMZ x Modell):")
        for variant, grp in overall.groupby("variant"):
            print(f"    {variant:<9} F1={grp['f1'].mean():.3f}  "
                  f"AUC={grp['roc_auc'].mean():.3f}")


if __name__ == "__main__":
    main()
