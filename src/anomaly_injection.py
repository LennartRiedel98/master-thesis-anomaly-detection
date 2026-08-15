"""Synthetische Anomalie-Injektion (gemeinsame Bibliothek fuer Stage 8/9).

Erzeugt kontrolliert kuenstliche Anomalien in einer stuendlichen
WMZ-Zeitreihe und liefert dazu eine punktgenaue Ground-Truth. Dieselbe
Maschinerie wird zweimal verwendet:

  - Stage 8 (HPO): Injektion in die *Validation*-Menge (eigener Seed),
    um unüberwachte Modelle anhand eines labelierten Signals zu tunen.
  - Stage 9 (Evaluation): Injektion in die *Test*-Menge (anderer Seed),
    als finales, waehrend HPO nie gesehenes Evaluations-Signal.

Anomalie-Taxonomie (Methodik 6.3 / 7.6) - bewusst disjunkt nach
Stationaritaet:

  Stationaer  (Varianten A/B, auf kW bzw. Residuum):
    spike     - kurzer Ausschlag nach oben (kW * Faktor, 1 h)
    drop       - Einbruch (kW * Faktor < 1, 1-3 h)
    plateau    - eingefrorener konstanter Wert (6-24 h)
    leakage    - additiver Offset (+x %, 48-168 h)

  Nicht-stationaer (Variante C, auf dem MSTL-Trend):
    drift             - linearer Anstieg (+x %/Tag)
    structural_break  - permanenter Niveausprung (+x %, bis Reihenende)

Designentscheidung zur Variante B (mit dem Betreuer abgestimmt): die
stationäre Störung wird als *dieselbe additive kW-Differenz* (gleiche
Zeitstempel, gleiche Intensität) auf das Residuum addiert. Das ist
konsistent mit der Stationaritaets-Argumentation aus Methodik 6.3
(Trend/Saison bleiben von stationären Anomalien unberührt), ohne pro
Injektion eine teure MSTL-Re-Dekomposition zu erfordern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# Anomalietypen je Schiene.
STATIONARY_TYPES = ["spike", "drop", "plateau", "leakage"]
NONSTATIONARY_TYPES = ["drift", "structural_break"]

# Intensitaeten je Typ. Die Evaluation (Stage 10) wertet pro
# (Typ x Intensitaet) getrennt aus, daher mehrere Stufen je Typ.
#   spike/drop : multiplikativer Faktor auf den kW-Wert
#   plateau    : None (haelt den Wert bei Event-Start konstant)
#   leakage    : additiver Anteil des kW-Werts (+15 % / +30 %)
#   drift      : Anstieg pro Tag als Anteil des Trend-Werts
#   structural_break : permanenter Anteil des Trend-Werts
DEFAULT_INTENSITIES: dict[str, list[float | None]] = {
    "spike": [3.0, 5.0, 10.0],
    "drop": [0.0, 0.25],
    "plateau": [None],
    "leakage": [0.15, 0.30],
    "drift": [0.005, 0.01],
    "structural_break": [0.10, 0.20],
}

# Dauer-Spannen in Stunden (uniform gezogen). drift/break enden am
# Reihenende bzw. dauern lange; ihre Dauer wird separat behandelt.
DURATION_HOURS: dict[str, tuple[int, int]] = {
    "spike": (1, 1),
    "drop": (1, 3),
    "plateau": (6, 24),
    "leakage": (48, 168),
    "drift": (14 * 24, 30 * 24),
}

# Anzahl Events je (Typ x Intensitaet) - klein gehalten, damit das
# Test-Fenster nicht ueberfuellt wird.
DEFAULT_EVENTS_PER_LEVEL = 4

# Mindestabstand zwischen Events (Stunden), damit Labels disjunkt bleiben.
MIN_GAP_H = 24

# Baseline-Schwelle: Events fuer multiplikative Typen (spike/drop) nur
# dort platzieren, wo das Signal aktiv ist. Verhindert das in Methodik 5.4
# beschriebene Problem, dass ein "Drop x0" im Sommer (Signal bereits ~0)
# nicht von der Grundlinie unterscheidbar ist.
ACTIVE_QUANTILE = 0.30


@dataclass
class AnomalyEvent:
    """Ein injiziertes Event: Typ, Intensitaet und Zeitfenster (Index-Positionen)."""
    kind: str
    intensity: float | None
    start: int            # inklusive Positions-Index
    end: int              # exklusive Positions-Index
    label: str = field(init=False)

    def __post_init__(self) -> None:
        """Eindeutiges Ground-Truth-Label aus Typ und Intensitaet bilden.

        Ergibt etwa ``spike@10`` oder - bei Typen ohne Intensitaetsstufe - nur
        ``plateau``. Dieses Label ist der Schluessel, ueber den Stage 10 die
        Ergebnisse nach Anomalietyp aufschluesselt.
        """
        # Eindeutiges Label fuer die Ground-Truth (Typ + Intensitaet).
        if self.intensity is None:
            self.label = self.kind
        else:
            self.label = f"{self.kind}@{self.intensity:g}"


# ---------------------------------------------------------------------------
# Event-Planung
# ---------------------------------------------------------------------------

def _placeable_positions(values: np.ndarray,
                         region: np.ndarray,
                         need_active: bool) -> np.ndarray:
    """Positions-Indizes innerhalb ``region``, an denen ein Event starten darf.

    ``region`` ist ein Boolean-Array (z. B. split == "test"). ``need_active``
    schraenkt auf Stunden mit Signal ueber dem ACTIVE_QUANTILE ein.
    """
    ok = region & ~np.isnan(values)
    if need_active:
        active = values[region & ~np.isnan(values)]
        if active.size:
            thr = np.quantile(active[active > 0], ACTIVE_QUANTILE) if (active > 0).any() else 0.0
            ok &= values > thr
    return np.flatnonzero(ok)


def plan_events(values: np.ndarray,
                region: np.ndarray,
                kinds: Iterable[str],
                rng: np.random.Generator,
                intensities: dict[str, list[float | None]] = None,
                events_per_level: int = DEFAULT_EVENTS_PER_LEVEL,
                ) -> list[AnomalyEvent]:
    """Plant nicht-ueberlappende Events fuer die gegebenen Typen.

    Greedy-Platzierung: pro (Typ, Intensitaet) werden ``events_per_level``
    Startpunkte zufaellig gezogen und nur akzeptiert, wenn ihr Fenster
    (inkl. MIN_GAP_H Puffer) noch frei ist. Bereits belegte Positionen
    werden in einer Belegungsmaske vermerkt.
    """
    intensities = intensities or DEFAULT_INTENSITIES
    n = len(values)
    occupied = np.zeros(n, dtype=bool)   # inkl. Puffer belegte Positionen
    events: list[AnomalyEvent] = []

    for kind in kinds:
        need_active = kind in ("spike", "drop")
        candidates = _placeable_positions(values, region, need_active)
        if candidates.size == 0:
            continue
        lo, hi = DURATION_HOURS.get(kind, (1, 1))

        for intensity in intensities.get(kind, [None]):
            placed = 0
            # Mehr Versuche als noetig, da viele Kandidaten kollidieren.
            for start in rng.permutation(candidates):
                if placed >= events_per_level:
                    break
                dur = int(rng.integers(lo, hi + 1))
                end = min(start + dur, n)
                # Fenster inkl. Puffer auf Ueberlappung pruefen.
                buf_lo = max(0, start - MIN_GAP_H)
                buf_hi = min(n, end + MIN_GAP_H)
                if occupied[buf_lo:buf_hi].any():
                    continue
                # Region-Grenzen respektieren (Event darf nicht aus dem
                # Ziel-Split herauslaufen).
                if not region[start:end].all():
                    continue
                occupied[buf_lo:buf_hi] = True
                events.append(AnomalyEvent(kind, intensity, start, int(end)))
                placed += 1

    return events


# ---------------------------------------------------------------------------
# Injektion
# ---------------------------------------------------------------------------

def compute_delta(values: np.ndarray,
                  events: list[AnomalyEvent]) -> np.ndarray:
    """Additive Differenz (in Signal-Einheiten), die die Events bewirken.

    Rueckgabe ist ein Array gleicher Laenge wie ``values`` mit 0 ueberall
    ausserhalb von Events. Fuer Varianten A und B wird genau dieses Delta
    auf das jeweilige Signal (kW bzw. Residuum) addiert - so ist die
    physikalische Stoerungs-Groesse in beiden Repraesentationen identisch.
    """
    delta = np.zeros(len(values), dtype=float)
    for ev in events:
        seg = values[ev.start:ev.end]
        base = np.nan_to_num(seg, nan=0.0)

        if ev.kind == "spike" or ev.kind == "drop":
            # Multiplikativ: neuer Wert = Faktor * alt -> Delta = (Faktor-1)*alt
            delta[ev.start:ev.end] = (ev.intensity - 1.0) * base
        elif ev.kind == "plateau":
            # Konstanter Wert (= erster Wert des Fensters) -> Delta gleicht aus.
            hold = base[0] if base.size else 0.0
            delta[ev.start:ev.end] = hold - base
        elif ev.kind == "leakage":
            # Additiver Offset von +intensity % des laufenden Werts.
            delta[ev.start:ev.end] = ev.intensity * base
        else:
            raise ValueError(f"compute_delta: unerwarteter stationärer Typ {ev.kind!r}")
    return delta


def compute_trend_delta(values: np.ndarray,
                        events: list[AnomalyEvent]) -> np.ndarray:
    """Additive Differenz fuer nicht-stationäre Trend-Anomalien (Variante C)."""
    delta = np.zeros(len(values), dtype=float)
    for ev in events:
        seg = values[ev.start:ev.end]
        base = np.nan_to_num(seg, nan=0.0)
        if ev.kind == "drift":
            # Linearer Anstieg: +intensity pro Tag (24 h) als Anteil des
            # Trend-Niveaus am Event-Start.
            hours = np.arange(ev.end - ev.start)
            ref = base[0] if base.size else 0.0
            delta[ev.start:ev.end] = ref * ev.intensity * (hours / 24.0)
        elif ev.kind == "structural_break":
            # Permanenter Niveausprung ab Event-Start bis Reihenende.
            ref = base[0] if base.size else 0.0
            delta[ev.start:] = ref * ev.intensity
        else:
            raise ValueError(f"compute_trend_delta: unerwarteter Typ {ev.kind!r}")
    return delta


def label_series(index: pd.DatetimeIndex,
                 events: list[AnomalyEvent]) -> pd.Series:
    """Ground-Truth: pro Zeitstempel das Event-Label (sonst NaN)."""
    labels = pd.Series(pd.NA, index=index, dtype="object")
    for ev in events:
        labels.iloc[ev.start:ev.end] = ev.label
    return labels


def kind_series(index: pd.DatetimeIndex,
                events: list[AnomalyEvent]) -> pd.Series:
    """Wie label_series, aber nur der Typ (ohne Intensitaet) - fuer die
    nach Anomalietyp stratifizierte Auswertung."""
    kinds = pd.Series(pd.NA, index=index, dtype="object")
    for ev in events:
        kinds.iloc[ev.start:ev.end] = ev.kind
    return kinds
