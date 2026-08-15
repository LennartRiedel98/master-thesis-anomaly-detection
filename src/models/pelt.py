"""PELT Change-Point-Detektor.

Pruned Exact Linear Time (Killick u. a. 2012) via ``ruptures``. Wird auf
den MSTL-Trend (Variante C) angewandt und liefert Change-Points im
langfristigen Verlauf. Ziel: nicht-stationaere Anomalien (Drift,
Strukturbruch, EnSikuMaV-September-2022).

Hyperparameter (vgl. Methodik 7.5):
    model:    "l2", "rbf", oder "linear" - Kostenfunktion
    penalty:  Strafterm (BIC-aehnlich); steuert Anzahl Change-Points
    min_size: minimaler Abstand zwischen Change-Points (in Stunden)

Anders als die anderen Detektoren liefert PELT diskrete Change-Point-
Positionen, nicht einen kontinuierlichen Score. Konvention fuer das
gemeinsame API:
    ``score(X)`` gibt eine binaere Serie zurueck (1 an Change-Point-
    Positionen, 0 sonst, NaN wo der Input NaN ist). Stage 10 behandelt
    PELT-Output entsprechend separat von den Score-basierten Detektoren.

PELT ist ein Offline-Verfahren: ``fit`` laeuft auf dem Trainings-Signal
und persistiert die dort gefundenen Change-Points als Artefakt; ``score``
fuehrt die Detektion auf dem jeweils uebergebenen Signal (z. B. dem
vollen Verlauf oder dem Test-Bereich) erneut aus.

``ruptures`` wird lazy importiert.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class PELTDetector(AnomalyDetector):
    """PELT: exakte Change-Point-Suche mit Straftermsteuerung (Killick 2012)."""

    name = "pelt"

    def __init__(self,
                 model: str = "l2",
                 penalty: float = 10.0,
                 min_size: int = 24 * 7,
                 **hparams: Any) -> None:
        """Hyperparameter setzen und die Change-Point-Liste leeren.

        ``min_size`` steht auf einer Woche (24*7 Stunden): Change-Points sollen
        Strukturbrueche im Betriebsniveau markieren, nicht den Tages- oder
        Wochenrhythmus. Ein kleinerer Wert wuerde die Saisonalitaet als Folge
        von Bruechen missdeuten.
        """
        super().__init__(model=model,
                         penalty=penalty,
                         min_size=min_size,
                         **hparams)
        self.feature_cols_: list[str] | None = None
        # Change-Point-Zeitstempel aus dem fit()-Signal (Artefakt).
        self.train_changepoints_: list[pd.Timestamp] = []

    def _detect(self, series: pd.Series) -> list[pd.Timestamp]:
        """Fuehrt PELT auf den nicht-fehlenden Werten von ``series`` aus.

        Gibt die Zeitstempel der Change-Points zurueck (ohne den von
        ruptures stets mitgelieferten Endpunkt n).
        """
        import ruptures as rpt

        valid = series.dropna()
        if len(valid) <= self.hparams["min_size"] * 2:
            return []
        signal = valid.to_numpy(dtype=float).reshape(-1, 1)
        algo = rpt.Pelt(model=self.hparams["model"],
                        min_size=self.hparams["min_size"]).fit(signal)
        bkps = algo.predict(pen=self.hparams["penalty"])
        # ruptures liefert End-Indizes der Segmente; der letzte ist len(n)
        # und kein echter Change-Point.
        cp_idx = [b for b in bkps if b < len(valid)]
        return [valid.index[i] for i in cp_idx]

    def fit(self, X: pd.DataFrame) -> "PELTDetector":
        """Change-Points auf dem Trainings-Signal suchen und merken.

        Erwartet genau eine Spalte - PELT arbeitet hier auf dem MSTL-Trend, also
        auf einem univariaten Signal. Die gefundenen Positionen werden als
        Artefakt behalten; fuer die Bewertung ist ``score`` zustaendig, das die
        Suche auf dem jeweils uebergebenen Abschnitt erneut ausfuehrt.
        """
        if X.shape[1] != 1:
            raise ValueError(
                f"PELTDetector erwartet ein univariates Signal, "
                f"bekam {X.shape[1]} Spalten: {list(X.columns)}"
            )
        self.feature_cols_ = list(X.columns)
        self.train_changepoints_ = self._detect(X.iloc[:, 0])
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Binaere Serie: 1 an Change-Points, 0 sonst, NaN bei fehlendem Wert.

        Das ist die Ausnahme von der Konvention 'hoeher = anomaler': PELT
        liefert keinen kontinuierlichen Score, sondern eine Entscheidung. Stage
        10 traegt dem Rechnung, indem es fuer PELT die Schwelle 0,5 fest setzt
        statt sie aus dem Validierungs-Quantil abzuleiten. Die ROC-AUC eines
        binaeren Ausgangs ist entsprechend nur eingeschraenkt aussagekraeftig.
        """
        self._check_fitted()
        series = X[self.feature_cols_[0]]
        out = pd.Series(np.nan, index=X.index, name="score")
        # 0 ueberall, wo ein Wert vorhanden ist; 1 an Change-Points.
        out.loc[series.notna()] = 0.0
        for ts in self._detect(series):
            out.loc[ts] = 1.0
        return out

    def save(self, path: Path) -> None:
        """Hyperparameter, Spaltenname und die Trainings-Change-Points ablegen."""
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "hparams": self.hparams,
                "feature_cols": self.feature_cols_,
                "train_changepoints": self.train_changepoints_,
            }, fh)

    @classmethod
    def load(cls, path: Path) -> "PELTDetector":
        """Instanz aus einer mit save() geschriebenen Datei rekonstruieren."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(**state["hparams"])
        obj.feature_cols_ = state["feature_cols"]
        obj.train_changepoints_ = state["train_changepoints"]
        obj.fitted = True
        return obj
