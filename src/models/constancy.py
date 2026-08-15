"""Konstanz-/Plateau-Detektor (kollektive Anomalie).

Motivation (Methodik § 6.6): Ein **Plateau** ist eine
*kollektive* Anomalie - jeder Einzelwert ist plausibel, nur die
**Konstanz** ueber ein Intervall ist anomal. Punkt- und rekonstruktions-
basierte Detektoren (Z-Score, LOF, IF, LSTM-AE) uebersehen das
systematisch (Recall ~ 0, vgl. § 4.4.2), weil kein einzelner Wert
auffaellt. Dieser Detektor schliesst genau diese Luecke und ist als
**typ-spezifische** Ergaenzung unter H1 (Methoden-Matching) gerahmt -
nicht als generischer Allrounder.

Verfahrensfamilie: Varianz-/Lauflaengen-basierte Konstanz-Erkennung,
verwandt mit Varianz-Change-Point (Inclán und Tiao 1994; Killick u. a. 2012),
Matrix Profile (Yeh u. a. 2016) und SAX-Motiven (Lin u. a. 2007). Bewusst **kein** Deep
Learning: rein statistisch, deterministisch, CPU-only (kein GPU-Training)
- damit on-the-edge tauglich (§ 5.5.3).

Score-Konstruktion ("hoeher = anomaler", API-Konvention base.py):

    score = flatness x collapse              (beide in [0, 1])

  * flatness = exp(-local_std / std_scale)
        local_std = rolling_std ueber das kurze Fenster ``window``.
        local_std -> 0  (Fenster ~ konstant)  => flatness -> 1.
    "Ist es JETZT nahezu konstant?"

  * collapse = 1 - exp(-max(base_std - local_std, 0) / (sensitivity*std_scale))
        base_std = rollierende Std ueber das lange, um ``window``
        zurueckliegende Kontextfenster ``baseline_window`` (juengste
        Eigen-Variabilitaet des Zaehlers VOR dem aktuellen Fenster).
    "Und ist das ein EINBRUCH der Variabilitaet gegenueber dem, was der
    Zaehler zuletzt selbst gezeigt hat?"

Diese **baseline-relative** Konstruktion operationalisiert den
geforderten Aktiv-Kontext ("konstant, *obwohl* aktiv erwartet") ueber die
**eigene Historie** des Zaehlers statt ueber ein absolutes Niveau oder
einen Kalender:

  * Anomales Plateau: Signal war aktiv/variabel -> friert ein. local_std
    bricht ein, base_std (Vor-Ereignis) ist hoch -> collapse gross,
    flatness gross -> hoher Score.
  * Legitime Off-/Standby-Flachphase (Sommer-Null wmz_3, Wochenend-
    Stillstand): Signal war *bereits* flach -> base_std ~ local_std ~ 0
    -> collapse ~ 0 -> Score ~ 0. Genau diese Phasen werden NICHT
    geflaggt, ohne dass ein absolutes Level-Gate noetig ist.
  * Normale aktive Variation: flatness ~ 0 -> Score ~ 0.

Ein absolutes Level-Gate (frueherer Entwurf) scheiterte an der WMZ-
Heterogenitaet: wmz_2 (Baseload) hat keinen Off-Zustand und friert
Plateaus im unteren Aktivband ein, wmz_3 hat eine grosse Off-Masse.
Der baseline-relative Ansatz braucht keine zaehler-spezifische Schwelle.

Variantenzuordnung: Der Detektor laeuft auf **Variante A (raw)** - dem
physikalischen kW-Signal, in dem ein Plateau tatsaechlich eine Konstante
ist. Auf dem MSTL-Residuum (Variante B) ist ein Plateau **nicht** flach
(konstantes kW minus variabler Trend/Saison = invertiertes Saison-
Muster), weshalb die Konstanz-Logik dort konzeptionell nicht greift; der
Detektor wird daher in der Registry nur fuer ``raw`` registriert. Er wird
gegen die **stationäre** Ground-Truth-Schiene bewertet (Plateau ist in
``STATIONARY_TYPES``).

Ehrliche Rahmung (in die Arbeit): Synthetische Plateaus sind perfekte
Konstanten -> ein hoher Recall ist teils tautologisch. Der eigentliche
Beleg liegt in der **Precision** auf den legitimen Flachphasen (haelt der
baseline-relative Kontext sie aus?) und in der Demonstration des Prinzips,
dass typ-spezifische Detektion die universelle Plateau-Luecke schliesst.

Hyperparameter (Stage-8-Grid):
    window:           kurzes Konstanz-Fenster (Stunden).
    baseline_window:  langes Kontextfenster fuer die Vor-Variabilitaet.
    sensitivity:      Skaliert, wie gross der Varianz-Einbruch (in std_scale-
                      Einheiten) fuer collapse -> 1 sein muss. Kleiner =
                      empfindlicher.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class ConstancyDetector(AnomalyDetector):
    """Konstanz-Detektor: meldet Plateaus, also unnatuerlich flache Abschnitte.

    Der einzige Eigenbau unter den sechs Detektoren. Er schliesst eine
    Luecke, die die uebrigen systematisch offen lassen (Recall nahe null
    fuer Plateaus): Bei einem Plateau ist jeder Einzelwert plausibel, nur
    die Konstanz ueber ein Intervall ist es nicht.
    """

    name = "constancy"

    def __init__(self,
                 window: int = 6,
                 baseline_window: int = 168,
                 sensitivity: float = 0.5,
                 **hparams: Any) -> None:
        """Hyperparameter pruefen und die spaeter gefitteten Felder anlegen.

        Die drei Pruefungen sind keine Formsache: ``window >= 2`` ist noetig,
        weil eine Standardabweichung ueber einen einzelnen Wert undefiniert ist;
        ``baseline_window > window`` stellt sicher, dass sich das Kontextfenster
        vom Messfenster ueberhaupt unterscheidet - waeren beide gleich, waere
        der Einbruch per Konstruktion immer null; ``sensitivity > 0`` verhindert
        eine Division durch null in der Einbruchs-Skala.
        """
        if window < 2:
            raise ValueError(f"window muss >= 2 sein, war {window}")
        if baseline_window <= window:
            raise ValueError(
                f"baseline_window ({baseline_window}) muss > window "
                f"({window}) sein.")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity muss > 0 sein, war {sensitivity}")
        super().__init__(window=window, baseline_window=baseline_window,
                         sensitivity=sensitivity, **hparams)
        # In fit() gesetzt:
        self.feature_cols_: list[str] | None = None
        self.signal_col_: str | None = None
        # Datengetriebene Skala (typische aktive Rolling-Std), nicht HP:
        self.std_scale_: float | None = None

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _pick_signal_col(self, cols: list[str]) -> str:
        """Waehlt die Signal-Spalte: bevorzugt das kW-Signal (Variante A),
        ersatzweise das Residuum. So bleibt der Detektor variantenagnostisch,
        auch wenn er regulaer nur fuer ``raw`` registriert ist."""
        for suffix in ("_kw_mean", "_residual"):
            hits = [c for c in cols if c.endswith(suffix)]
            if hits:
                # model_features schneidet pro WMZ -> genau eine passende Spalte.
                return hits[0]
        raise ValueError(
            "ConstancyDetector findet keine Signal-Spalte "
            f"(_kw_mean/_residual) in {cols}"
        )

    def _local_std(self, s: pd.Series) -> pd.Series:
        """Standardabweichung im kurzen Messfenster - wie flach ist es *jetzt*?

        ``min_periods`` laesst halb gefuellte Fenster zu, damit am Reihenanfang
        und nach Luecken nicht unnoetig viele NaN entstehen.
        """
        w = int(self.hparams["window"])
        return s.rolling(w, min_periods=max(2, w // 2)).std()

    def _base_std(self, s: pd.Series) -> pd.Series:
        """Variabilitaet im langen Kontextfenster, um ``window`` Stunden
        zurueckversetzt - so spiegelt sie am Plateau-Beginn die VOR dem
        Ereignis liegende Aktivitaet, nicht das einfrierende Fenster selbst."""
        w = int(self.hparams["window"])
        bw = int(self.hparams["baseline_window"])
        return (s.shift(w)
                 .rolling(bw, min_periods=max(2, bw // 4))
                 .std())

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "ConstancyDetector":
        """Die datengetriebene Streuungsskala aus den Trainingsdaten bestimmen.

        Gelernt wird genau eine Zahl: die typische (mediane) Rolling-Streuung im
        aktiven Betrieb dieses Zaehlers. Sie ist der Massstab, gegen den spaeter
        sowohl 'flach' als auch 'eingebrochen' gemessen wird - und der Grund,
        warum derselbe Detektor ohne Nachjustierung an einem Zaehler mit ganz
        anderem Lastniveau funktioniert. Der Median ist bewusst gewaehlt, weil
        er gegen die im Training verbliebenen Ausreisser robust ist; der Floor
        faengt den Grenzfall eines durchgehend konstanten Signals ab.
        """
        self.feature_cols_ = list(X.columns)
        self.signal_col_ = self._pick_signal_col(self.feature_cols_)

        s = X[self.signal_col_].dropna()

        # std_scale: typische (mediane) aktive Rolling-Std des sauberen
        # Trainings. Kalibriert sowohl die Glaette-Schwelle (flatness) als
        # auch die Einbruchs-Skala (collapse) relativ zur normalen
        # Schwankung dieses Zaehlers. Floor gegen Division durch ~0.
        rstd = self._local_std(s).dropna()
        rstd = rstd[rstd > 0]
        median_rstd = float(rstd.median()) if len(rstd) else 0.0
        floor = 1e-6 * (float(s.std(ddof=0)) or 1.0)
        self.std_scale_ = max(median_rstd, floor)

        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Plateau-Score je Stunde als Produkt aus Flachheit und Einbruch.

        Die beiden Faktoren muessen *beide* zutreffen, deshalb das Produkt:

        - ``flatness`` misst, ob das aktuelle Fenster nahezu konstant ist. Fuer
          sich genommen wuerde es jede Nachtabsenkung und jeden Stillstand
          melden - beides ist normal.
        - ``collapse`` misst, ob die Schwankung gegenueber der juengsten eigenen
          Historie eingebrochen ist. Das Kontextfenster ist um ``window``
          zurueckversetzt, sieht also den Zustand *vor* dem Plateau.

        Erst gemeinsam ergeben sie die gesuchte Aussage 'konstant, obwohl hier
        Aktivitaet zu erwarten waere'. Genau diese kollektive Anomalie uebersehen
        die punktbasierten Detektoren systematisch, weil kein einzelner Wert
        auffaellt.

        Am Reihenanfang, wo noch kein Kontext existiert, wird 0 statt NaN
        gesetzt: Fehlender Kontext ist kein Anomaliehinweis.
        """
        self._check_fitted()
        s = X[self.signal_col_]

        local_std = self._local_std(s)
        base_std = self._base_std(s)

        # flatness: ist das Fenster JETZT nahezu konstant?
        flatness = np.exp(-local_std / self.std_scale_)

        # collapse: ist die Variabilitaet gegenueber der juengsten Eigen-
        # Historie eingebrochen? Nur positive Einbrueche zaehlen; ein
        # bereits flaches Off-Fenster (base_std ~ local_std) -> collapse ~ 0.
        drop = (base_std - local_std).clip(lower=0.0)
        scale = self.hparams["sensitivity"] * self.std_scale_
        collapse = 1.0 - np.exp(-drop / scale)

        raw = flatness * collapse

        # API-Konvention: NaN-Signal -> NaN-Score. Wo base_std am Reihen-
        # anfang noch undefiniert ist, gibt es keinen Kontext -> kein
        # Anomalie-Signal (Score 0), sofern das Signal selbst vorhanden ist.
        out = pd.Series(raw, index=X.index, name="score")
        out[base_std.isna()] = 0.0
        out[s.isna()] = np.nan
        return out

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Hyperparameter, Spaltennamen, Signalspalte und Streuungsskala ablegen."""
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "hparams": self.hparams,
                "feature_cols": self.feature_cols_,
                "signal_col": self.signal_col_,
                "std_scale": self.std_scale_,
            }, fh)

    @classmethod
    def load(cls, path: Path) -> "ConstancyDetector":
        """Instanz aus einer mit save() geschriebenen Datei rekonstruieren."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(**state["hparams"])
        obj.feature_cols_ = state["feature_cols"]
        obj.signal_col_ = state["signal_col"]
        obj.std_scale_ = state["std_scale"]
        obj.fitted = True
        return obj
