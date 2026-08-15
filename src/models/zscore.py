"""Z-Score Anomalie-Detektor.

Konzept: per-Feature Z-Score gegen Train-Mean und Train-Std. Score ist
eine Aggregation von ``|z|`` ueber die Feature-Dimensionen. Nach
Stage-6-Normalisierung sind Mean/Std bereits ~0/~1 auf den Train-Zeilen,
der Z-Score reduziert sich dort effektiv auf den absoluten Featurewert.
Wir berechnen Mean/Std hier dennoch erneut auf den uebergebenen
Trainings-Zeilen, damit der Detektor self-contained und unabhaengig von
der vorgelagerten Skalierung korrekt bleibt.

Hyperparameter:
    aggregation: "max" | "l2" | "mean"  - wie mehrere Feature-Dimensionen
                 zu einem Score reduziert werden.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class ZScoreDetector(AnomalyDetector):
    """Z-Score-Detektor: Abstand zum Trainingsmittel in Standardabweichungen."""

    name = "zscore"

    def __init__(self, aggregation: str = "max", **hparams: Any) -> None:
        """Aggregationsart pruefen und die spaeter gefitteten Felder anlegen."""
        if aggregation not in ("max", "l2", "mean"):
            raise ValueError(f"Unbekannte aggregation: {aggregation!r}")
        super().__init__(aggregation=aggregation, **hparams)
        # Wird in fit() gesetzt:
        self.feature_cols_: list[str] | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "ZScoreDetector":
        """Mittelwert und Standardabweichung je Feature auf den Train-Zeilen.

        Mehr passiert nicht - der Detektor hat kein Modell im eigentlichen
        Sinn, nur zwei Vektoren. Genau das macht ihn zur Referenzlinie: Was
        ein aufwendigeres Verfahren gegen ihn nicht gewinnt, hat seinen
        Mehraufwand nicht gerechtfertigt.
        """
        # Nur vollstaendige Zeilen sehen - sonst zieht ein einzelnes NaN
        # die Spaltenstatistik (mit skipna zwar nicht, aber wir bleiben
        # konsistent zur score()-Maske und vermeiden Verzerrungen durch
        # teil-fehlende Zeilen).
        complete = X.dropna(axis=0, how="any")
        self.feature_cols_ = list(X.columns)
        self.mean_ = complete.mean()
        std = complete.std(ddof=0)
        # Konstante Spalten: std=0 -> Division durch Null vermeiden.
        self.std_ = std.where(std > 0, 1.0)
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Je Zeile den aggregierten Betrag der Z-Werte zurueckgeben.

        Die Aggregation ueber die Feature-Dimensionen ist der einzige
        Hyperparameter: ``max`` schlaegt an, sobald *ein* Feature auffaellt
        (empfindlich fuer Punktanomalien), ``mean`` verlangt eine breite
        Auffaelligkeit, ``l2`` liegt dazwischen.
        """
        self._check_fitted()
        cols = self.feature_cols_
        z = (X[cols] - self.mean_) / self.std_
        absz = z.abs()

        agg = self.hparams["aggregation"]
        if agg == "max":
            raw = absz.max(axis=1)
        elif agg == "mean":
            raw = absz.mean(axis=1)
        else:  # "l2"
            raw = np.sqrt((absz ** 2).sum(axis=1))

        # Zeilen mit irgendeinem NaN-Feature -> NaN-Score (API-Konvention).
        valid = X[cols].notna().all(axis=1)
        out = pd.Series(np.nan, index=X.index, name="score")
        out.loc[valid] = raw.loc[valid]
        return out

    def save(self, path: Path) -> None:
        """Hyperparameter, Spaltennamen, Mittel und Streuung als Pickle ablegen."""
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "hparams": self.hparams,
                "feature_cols": self.feature_cols_,
                "mean": self.mean_,
                "std": self.std_,
            }, fh)

    @classmethod
    def load(cls, path: Path) -> "ZScoreDetector":
        """Instanz aus einer mit save() geschriebenen Datei rekonstruieren."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(**state["hparams"])
        obj.feature_cols_ = state["feature_cols"]
        obj.mean_ = state["mean"]
        obj.std_ = state["std"]
        obj.fitted = True
        return obj
