"""Evaluations-Metriken fuer die Anomaliedetektion (Stage 8/10).

Stellt die in Methodik 10a beschriebenen Metriken bereit:
Precision/Recall/F1 mit **Point-Adjust** (Xu u. a. 2018) fuer
Intervall-Anomalien sowie threshold-freie ROC-AUC.

Point-Adjust-Konvention: gilt ein einziger Punkt innerhalb eines wahren
Anomalie-Segments als detektiert, wird das *gesamte* Segment als
detektiert gewertet. Das ist in der Zeitreihen-AD ueblich, ueberschaetzt
die Leistung aber bekanntermassen (Kim u. a. 2022) - in der Arbeit
entsprechend zu diskutieren. Wir berichten daher zusaetzlich die
threshold-freien ROC-AUC und PR-AUC auf Punkt-Ebene; PR-AUC ist bei
seltenen Anomalien trennschaerfer (Davis und Goadrich 2006; Saito und
Rehmsmeier 2015).

Schwellwerte werden datengetrieben aus einem Quantil der (sauberen)
Validation-Scores gewaehlt, nicht vom Modell - so bleibt die
Schwellwert-Wahl unabhaengig vom Test-Set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Zusammenhaengende True-Laeufe in einer Boolean-Maske als (start, end)."""
    if mask.size == 0:
        return []
    # Differenz der zu int gecasteten Maske: +1 = Lauf-Start, -1 = Lauf-Ende.
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Point-Adjust: markiert ganze wahre Segmente als erkannt, wenn darin
    mindestens ein Punkt positiv vorhergesagt wurde."""
    adjusted = y_pred.copy()
    for start, end in segments(y_true):
        if y_pred[start:end].any():
            adjusted[start:end] = True
    return adjusted


def threshold_from_quantile(scores: pd.Series, q: float) -> float:
    """Schwellwert = q-Quantil der (sauberen) Scores. NaN wird ignoriert."""
    return float(np.nanquantile(scores.to_numpy(dtype=float), q))


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, Recall, F1 aus binaeren Arrays."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_auc(scores: pd.Series, y_true: np.ndarray) -> float:
    """ROC-AUC auf Punkt-Ebene; NaN-Scores werden ausgeschlossen.

    Gibt NaN zurueck, wenn nur eine Klasse vorhanden ist (AUC undefiniert).
    """
    s = scores.to_numpy(dtype=float)
    valid = ~np.isnan(s)
    yt = y_true[valid]
    if yt.all() or not yt.any():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(yt, s[valid]))


def pr_auc(scores: pd.Series, y_true: np.ndarray) -> float:
    """Average Precision = Flaeche unter der Precision-Recall-Kurve (Punkt-Ebene).

    Bei seltenen Anomalien aussagekraeftiger als ROC-AUC: PR-AUC fokussiert
    die (seltene) Positiv-Klasse, waehrend ROC-AUC ueber beide Klassen
    mittelt und bei starkem Ungleichgewicht truegerisch flach/hoch wird
    (Davis und Goadrich 2006; Saito und Rehmsmeier 2015). NaN-Scores werden
    ausgeschlossen; bei nur einer Klasse -> NaN (undefiniert).
    """
    s = scores.to_numpy(dtype=float)
    valid = ~np.isnan(s)
    yt = y_true[valid]
    if yt.all() or not yt.any():
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(yt, s[valid]))


def evaluate(scores: pd.Series,
             y_true: pd.Series,
             threshold: float,
             adjust: bool = True) -> dict[str, float]:
    """Komplette Metrik-Auswertung fuer eine Score-Serie gegen binaere GT.

    ``scores`` und ``y_true`` teilen denselben Index. NaN-Scores
    (z. B. Luecken oder Fenster-Anlauf beim LSTM) zaehlen als
    "nicht detektiert" (y_pred=False), gehen aber nicht in die threshold-
    freien ROC-AUC/PR-AUC ein.

    Hinweis: ROC-AUC ist nur fuer kontinuierliche Scores aussagekraeftig;
    fuer binaere Detektoren (PELT) ist sie - wie PR-AUC - degeneriert und
    nur der Vollstaendigkeit halber enthalten (Beurteilung ueber F1).
    """
    s = scores.reindex(y_true.index)
    yt = y_true.to_numpy(dtype=bool)

    y_pred = (s.to_numpy(dtype=float) > threshold)
    y_pred = np.where(np.isnan(s.to_numpy(dtype=float)), False, y_pred).astype(bool)

    if adjust:
        y_pred_eval = point_adjust(yt, y_pred)
    else:
        y_pred_eval = y_pred

    precision, recall, f1 = _prf(yt, y_pred_eval)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(s, yt),
        "pr_auc": pr_auc(s, yt),
        "n_true": int(yt.sum()),
        "n_pred": int(y_pred.sum()),
        "threshold": threshold,
    }
