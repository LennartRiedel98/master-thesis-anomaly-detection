"""Variant -> Detektor-Klassen + Feature-Slicing pro WMZ.

Eine zentrale Stelle, an der die Pipeline-Architektur aus Methodik-
Sektion 6.2 in Code festgehalten ist:

    raw       (Variante A): Z-Score, LOF, IF, LSTM-AE, Constancy
    residual  (Variante B): Z-Score, LOF, IF, LSTM-AE
    trend     (Variante C): PELT, LSTM-AE

Pro Variante existiert ausserdem eine Slicing-Vorschrift, die aus der
normalisierten Feature-Tabelle (Output von Stage 6) die Spalten heraus-
zieht, die ein Modell fuer einen bestimmten WMZ sieht. Geteilte
Features (Zeit, Wetter) werden allen WMZ-Schnitten beigemischt.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base import AnomalyDetector
from .constancy import ConstancyDetector
from .iforest import IForestDetector
from .lof import LOFDetector
from .lstm_ae import LSTMAutoencoderDetector
from .pelt import PELTDetector
from .zscore import ZScoreDetector

WMZ_NAMES = ["wmz_1", "wmz_2", "wmz_3"]

# Variant -> Modell-Klassen. Aenderungen hier (z. B. Anomaly Transformer
# ergaenzen) propagieren automatisch in Stage 7.
#
# Variante A (raw) und B (residual) bekommen dieselben vier Modelle -
# das ist die Ablation aus Methodik 6.2: gleiche Modelle, unter-
# schiedlicher Input. Daraus laesst sich die Wirkung der STL-
# Vorverarbeitung empirisch ablesen.
#
# Variante C (trend) ist strukturell anders: PELT findet Change-Points,
# LSTM-AE auf dem Trend dient als Vergleichsbaseline gegen PELT.
#
# ConstancyDetector laeuft NUR auf Variante A (raw): Er prueft die Konstanz
# des physikalischen kW-Signals (Plateau = echte Konstante). Auf dem
# MSTL-Residuum (Variante B) ist ein Plateau nicht flach (konstantes kW
# minus variabler Saison = invertiertes Saison-Muster), daher dort nicht
# registriert. Er schliesst die universelle Plateau-Luecke (Recall ~ 0 bei
# allen punkt-/rekonstruktionsbasierten Modellen, § 4.4.2) als
# typ-spezifische Ergaenzung unter H1.
REGISTRY: dict[str, list[type[AnomalyDetector]]] = {
    "raw":      [ZScoreDetector, LOFDetector, IForestDetector,
                 LSTMAutoencoderDetector, ConstancyDetector],
    "residual": [ZScoreDetector, LOFDetector, IForestDetector, LSTMAutoencoderDetector],
    "trend":    [PELTDetector, LSTMAutoencoderDetector],
}


def _per_wmz_cols(df_cols: Iterable[str], wmz: str) -> list[str]:
    """Spalten, die zu genau einem WMZ gehoeren - Praefix-Match.

    Meta-Spalten (was_flagged, interpolated) sowie die Stage-5-Spalte
    ``split`` werden hier nicht gefiltert (die liegen ohnehin nicht im
    Input des Modells - Stage 7 trennt sie separat).
    """
    return [c for c in df_cols if c.startswith(f"{wmz}_")]


def model_features(df: pd.DataFrame, variant: str, wmz: str) -> pd.DataFrame:
    """Slice der normalisierten Feature-Tabelle fuer (variant, wmz).

    Erwartet df = Output aus Stage 6 (stage6_normalized_<variant>.parquet)
    inklusive split-Spalte und Meta-Spalten - diese werden hier
    ausgeschlossen.

    Beispiel fuer Variante "raw" und wmz_2:
        per_wmz = ["wmz_2_kw_mean", "wmz_2_rolling_mean_6h", ...]
        andere WMZ ("wmz_1_*", "wmz_3_*") werden ausgeschlossen
        shared  = ["hour_sin", "hour_cos", ..., "temperature", "humidity"]
        Rueckgabe: per_wmz + shared (per_wmz zuerst, sodass die ersten
        Spalten die WMZ-spezifischen sind - hilft bei Plots und Debugging)
    """
    if variant not in REGISTRY:
        raise ValueError(f"Unbekannte Variante: {variant!r}")

    # Meta- und Split-Spalten sind nie Modell-Input.
    drop = {"split"}
    drop.update(c for c in df.columns if c.endswith("_was_flagged"))
    drop.update(c for c in df.columns if c.endswith("_interpolated"))
    feature_cols = [c for c in df.columns if c not in drop]
    feat = df[feature_cols]

    if variant == "trend":
        # Variante C ist univariat: genau eine Spalte pro WMZ.
        # PELT erwartet eine 1D-Reihe; LSTM-AE behandelt sie als Sequenz.
        return feat[[f"{wmz}_trend"]]

    # Per-WMZ-Spalten = alle Spalten mit Praefix "wmz_2_" (Beispiel).
    per_wmz = _per_wmz_cols(feat.columns, wmz)

    # Die anderen WMZs aus der "shared"-Liste herausfiltern, sonst saehe
    # ein Modell fuer wmz_2 auch die Spalten von wmz_1 und wmz_3 - das
    # waere Cross-Leakage zwischen Zaehlern und wuerde die Per-WMZ-
    # Architektur unterlaufen.
    other_wmz = {f"wmz_{n}" for n in (1, 2, 3)} - {wmz}
    shared = [c for c in feat.columns
              if c not in per_wmz
              and not any(c.startswith(f"{ow}_") for ow in other_wmz)]
    return feat[per_wmz + shared]


def iter_training_jobs() -> Iterable[tuple[str, str, type[AnomalyDetector]]]:
    """Yields alle (variant, wmz, model_cls)-Kombinationen, die in
    Stage 7 trainiert werden sollen. Reihenfolge ist deterministisch."""
    for variant, model_classes in REGISTRY.items():
        for wmz in WMZ_NAMES:
            for model_cls in model_classes:
                yield variant, wmz, model_cls
