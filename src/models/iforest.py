"""Isolation Forest Detektor.

sklearn.ensemble.IsolationForest. ``score_samples`` gibt einen
Anomaly-Score, bei dem groessere (weniger negative) Werte normaler sind;
wir invertieren das Vorzeichen, sodass groesser = anomaler.

Hyperparameter (vgl. Methodik 7.5):
    n_estimators:  z. B. 200
    contamination: 'auto' oder kleiner Float
    max_features:  Float in (0, 1] oder int
    random_state:  fuer Reproduzierbarkeit

``sklearn`` wird lazy importiert (siehe lof.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class IForestDetector(AnomalyDetector):
    """Isolation Forest: Anomalien lassen sich mit wenigen Schnitten abtrennen.

    Der Gesamt-Champion dieser Arbeit (bestes F1 ueber alle Jobs).
    """

    name = "iforest"

    def __init__(self,
                 n_estimators: int = 200,
                 contamination: float | str = "auto",
                 max_features: float = 1.0,
                 random_state: int = 42,
                 **hparams: Any) -> None:
        """Hyperparameter setzen und die spaeter gefitteten Felder anlegen.

        ``random_state`` ist bewusst fest vorbelegt: Der IForest zieht seine
        Teilstichproben und Schnittachsen zufaellig, ohne festen Startwert
        waeren die Ergebnisse zwischen zwei Laeufen nicht identisch.
        """
        super().__init__(n_estimators=n_estimators,
                         contamination=contamination,
                         max_features=max_features,
                         random_state=random_state,
                         **hparams)
        self.feature_cols_: list[str] | None = None
        self._model = None

    def fit(self, X: pd.DataFrame) -> "IForestDetector":
        """Isolation Forest auf den vollstaendigen Train-Zeilen aufbauen.

        Der Wald besteht aus zufaelligen Schnitten durch den Merkmalsraum. Ein
        Punkt, der sich schon nach wenigen Schnitten allein in einem Blatt
        wiederfindet, liegt abseits der Masse - genau das ist das Anomaliemass.
        Anders als LOF braucht das Verfahren dafuer keinen Dichtebegriff und
        skaliert deshalb gut in hoeheren Dimensionen.
        """
        # Lazy-Import (siehe Modul-Docstring): sklearn erst zur Laufzeit
        # holen, damit models.registry auch ohne sklearn importierbar bleibt.
        from sklearn.ensemble import IsolationForest

        # Nur vollstaendige Zeilen trainieren - der IForest kann mit NaN
        # nicht umgehen, und Luecken/interpolierte Stunden sollen das
        # Normalmodell nicht verfaelschen (API-Konvention aus base.py).
        complete = X.dropna(axis=0, how="any")
        if complete.empty:
            raise ValueError("IForestDetector.fit: keine vollstaendigen Zeilen.")
        # Spaltenreihenfolge merken, damit score() exakt dieselben Features
        # in derselben Reihenfolge an das Modell gibt.
        self.feature_cols_ = list(X.columns)
        self._model = IsolationForest(
            n_estimators=self.hparams["n_estimators"],
            contamination=self.hparams["contamination"],
            max_features=self.hparams["max_features"],
            random_state=self.hparams["random_state"],
        )
        self._model.fit(complete.to_numpy())
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Negierten Isolations-Score je Zeile zurueckgeben (gross = anomal).

        ``score_samples`` liefert groessere (weniger negative) Werte fuer
        *normalere* Punkte; das Vorzeichen wird gedreht, damit die Konvention
        aus ``base.py`` gilt. Zeilen mit fehlenden Features bekommen NaN.
        """
        self._check_fitted()
        cols = self.feature_cols_
        # Zeilen mit irgendeinem NaN-Feature koennen nicht gescored werden;
        # sie bekommen NaN (base.py-Konvention) und werden hier maskiert,
        # damit der Output-Index exakt zu X passt.
        valid = X[cols].notna().all(axis=1)
        out = pd.Series(np.nan, index=X.index, name="score")
        if valid.any():
            # score_samples: groesser (weniger negativ) = normaler. Wir
            # negieren, damit - wie bei allen Detektoren - groesser = anomaler.
            raw = self._model.score_samples(X.loc[valid, cols].to_numpy())
            out.loc[valid] = -raw
        return out

    def save(self, path: Path) -> None:
        """Hyperparameter, Spaltennamen und den gefitteten Wald als Pickle ablegen."""
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "hparams": self.hparams,
                "feature_cols": self.feature_cols_,
                "model": self._model,
            }, fh)

    @classmethod
    def load(cls, path: Path) -> "IForestDetector":
        """Instanz aus einer mit save() geschriebenen Datei rekonstruieren."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(**state["hparams"])
        obj.feature_cols_ = state["feature_cols"]
        obj._model = state["model"]
        obj.fitted = True
        return obj
