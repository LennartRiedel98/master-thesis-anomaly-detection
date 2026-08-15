"""Gemeinsame Schnittstelle aller Anomalie-Detektoren (Stages 7-10).

Designziel: eine einheitliche Programmierschnittstelle, damit die
Trainings-Orchestrierung in Stage 7 alle Modelle gleich behandeln kann.
Ein neuer Detektor braucht deshalb nur diese vier Methoden und einen
Eintrag in ``registry.py`` - an der Pipeline selbst aendert sich nichts.

Konvention "hoeher = anomaler":
    ``score(X)`` gibt fuer jede Zeile einen Anomalie-Score zurueck.
    Groessere Werte bedeuten staerker abweichend. Schwellwerte werden
    in Stage 10 datengetrieben (per Validation-Quantil) gewaehlt, nicht
    vom Modell.

NaN-Konvention:
    Zeilen mit NaN im Input liefern NaN als Score. ``fit`` muss NaN-
    Zeilen selbst ausschliessen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class AnomalyDetector(ABC):
    """Gemeinsame Schnittstelle aller Detektoren.

    Subklassen setzen ``name`` als Klassen-Attribut und implementieren
    ``fit`` / ``score`` / ``save`` / ``load``. ABC heisst hier:
    Python verweigert die Instanziierung dieser Klasse direkt - es muss
    eine konkrete Subklasse sein, die alle ``@abstractmethod``s gefuellt
    hat.
    """

    # Default fuer die abstrakte Klasse selbst. Subklassen ueberschreiben
    # mit z. B. ``name = "zscore"``. Wird in Pfaden (models/<variant>/
    # <wmz>/<name>.pkl) und in Log-Ausgaben verwendet.
    name: str = "abstract"

    def __init__(self, **hparams: Any) -> None:
        """Hyperparameter uebernehmen und den Trainingsstatus zuruecksetzen."""
        # Hyperparameter werden als dict gespeichert, sodass save/load
        # sie problemlos serialisieren koennen. Subklassen reichen ihre
        # eigenen HPs via super().__init__(...) hierher.
        self.hparams: dict[str, Any] = dict(hparams)
        # Trainingsstatus-Flag. Wird in fit() auf True gesetzt; score()
        # ruft _check_fitted(), das den Aufruf vor fit() ablehnt.
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        """Trainiert auf vollstaendigen Zeilen von X. Gibt self zurueck."""

    @abstractmethod
    def score(self, X: pd.DataFrame) -> pd.Series:
        """Anomalie-Score pro Zeile, indexiert wie X. NaN-Zeilen -> NaN."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persistiert Modellzustand nach ``path`` (Verzeichnis oder Datei)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        """Laedt persistierten Zustand und rekonstruiert die Instanz."""

    # ------------------------------------------------------------------
    # Hilfsmethoden (von allen konkreten Detektoren geteilt)
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Abbrechen, wenn score() vor fit() gerufen wird.

        Ohne diese Pruefung wuerden die Detektoren mit einem schwer
        lesbaren AttributeError auf ein fehlendes internes Feld laufen.
        """
        if not self.fitted:
            raise RuntimeError(
                f"{type(self).__name__}.score() vor fit() aufgerufen."
            )

    def __repr__(self) -> str:
        """Klassenname mit allen Hyperparametern - so wie im Log gebraucht."""
        hp = ", ".join(f"{k}={v!r}" for k, v in self.hparams.items())
        return f"{type(self).__name__}({hp})"
