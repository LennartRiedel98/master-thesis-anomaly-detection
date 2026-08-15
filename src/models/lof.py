"""Local Outlier Factor Detektor.

sklearn.neighbors.LocalOutlierFactor mit ``novelty=True``, damit nach dem
Fit auf Train neue Punkte gescored werden koennen. ``score_samples``
liefert das Negierte des LOF (groesser = normaler); wir invertieren das
Vorzeichen, sodass groesser = anomaler (API-Konvention aus base.py).

Hyperparameter (vgl. Methodik 7.5):
    n_neighbors:   z. B. 20
    contamination: 'auto' oder kleiner Float (nur fuer das interne
                   Schwellwert-Attribut relevant; unsere Scores sind
                   schwellwert-frei und werden in Stage 10 quantilbasiert
                   geschnitten).

``sklearn`` wird lazy importiert, damit ``models.registry`` auch auf
Maschinen ohne sklearn-Installation importierbar bleibt.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class LOFDetector(AnomalyDetector):
    """Local Outlier Factor: Dichtevergleich mit der eigenen Nachbarschaft."""

    name = "lof"

    def __init__(self,
                 n_neighbors: int = 20,
                 contamination: float | str = "auto",
                 **hparams: Any) -> None:
        """Hyperparameter setzen und die spaeter gefitteten Felder anlegen.

        ``n_neighbors`` ist der eigentliche Stellhebel: Er legt fest, wie gross
        die Nachbarschaft ist, gegen deren Dichte ein Punkt verglichen wird. Zu
        klein macht den Detektor rauschempfindlich, zu gross verwischt lokale
        Strukturen - Stage 8 sucht den Wert per Gittersuche.
        """
        super().__init__(n_neighbors=n_neighbors,
                         contamination=contamination,
                         **hparams)
        self.feature_cols_: list[str] | None = None
        self._model = None   # sklearn-Objekt, lazy erzeugt

    def fit(self, X: pd.DataFrame) -> "LOFDetector":
        """Nachbarschaftsmodell auf den vollstaendigen Train-Zeilen aufbauen.

        ``novelty=True`` ist zwingend: Nur in diesem Modus laesst sich das
        gefittete Modell spaeter auf *neue* Punkte anwenden. Im Standardmodus
        koennte LOF ausschliesslich die Trainingsdaten selbst bewerten, was
        fuer eine Train/Test-Trennung unbrauchbar waere.
        """
        from sklearn.neighbors import LocalOutlierFactor

        complete = X.dropna(axis=0, how="any")
        if complete.empty:
            raise ValueError("LOFDetector.fit: keine vollstaendigen Zeilen.")
        self.feature_cols_ = list(X.columns)
        self._model = LocalOutlierFactor(
            n_neighbors=self.hparams["n_neighbors"],
            contamination=self.hparams["contamination"],
            novelty=True,
        )
        self._model.fit(complete.to_numpy())
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Negierten LOF je Zeile zurueckgeben (gross = anomal).

        ``score_samples`` liefert groessere Werte fuer *normalere* Punkte;
        das Vorzeichen wird gedreht, damit die Konvention aus ``base.py``
        gilt. Zeilen mit fehlenden Features bekommen NaN.
        """
        self._check_fitted()
        cols = self.feature_cols_
        valid = X[cols].notna().all(axis=1)
        out = pd.Series(np.nan, index=X.index, name="score")
        if valid.any():
            # score_samples: groesser = normaler -> negieren.
            raw = self._model.score_samples(X.loc[valid, cols].to_numpy())
            out.loc[valid] = -raw
        return out

    def save(self, path: Path) -> None:
        """Hyperparameter, Spaltennamen und das sklearn-Modell als Pickle ablegen."""
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "hparams": self.hparams,
                "feature_cols": self.feature_cols_,
                "model": self._model,
            }, fh)

    @classmethod
    def load(cls, path: Path) -> "LOFDetector":
        """Instanz aus einer mit save() geschriebenen Datei rekonstruieren."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(**state["hparams"])
        obj.feature_cols_ = state["feature_cols"]
        obj._model = state["model"]
        obj.fitted = True
        return obj
