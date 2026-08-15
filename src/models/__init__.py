"""Anomalie-Detektoren fuer Stages 7-10.

Jedes Modell implementiert das ``AnomalyDetector``-Protokoll aus
``base.py``. Die Zuordnung Variante -> Modell-Klassen lebt in
``registry.py``.

Implementiert sind alle sechs Detektoren der Arbeit: ``zscore``, ``lof``,
``iforest``, ``constancy`` (Schiene A/raw bzw. A+B), ``pelt`` (Schiene
C/trend) und ``lstm_ae`` (alle drei Schienen).
"""

from .base import AnomalyDetector

__all__ = ["AnomalyDetector"]
