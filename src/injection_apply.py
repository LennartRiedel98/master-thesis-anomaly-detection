"""Injektion auf Feature-Ebene anwenden (gemeinsam fuer Stage 8 und 9).

Nimmt die *unnormalisierten* Stage-4-Features plus den Stage-3-Trend,
injiziert Anomalien in eine Ziel-Region (Validation bzw. Test), berechnet
die abgeleiteten Features (Rolling-Statistiken) auf dem gestoerten Signal
neu und normalisiert anschliessend mit dem in Stage 6 auf Train gefitteten
Scaler (reines transform - kein Leakage).

Damit reicht die Stoerung physikalisch korrekt bis in die Rolling-Features
durch (ein Spike erhoeht z. B. auch das rollierende Maximum), und die
Skalierung bleibt exakt die der Trainingsverteilung.

Rueckgabe:
    frames : dict variant -> normalisierte, injizierte Feature-Tabelle
             (volle Reihe; der Aufrufer schneidet die Region selbst zu)
    gt     : DataFrame mit Ground-Truth-Spalten je WMZ:
                gt_stat_<wmz>        Anomalietyp (stationär) oder NA
                gt_stat_label_<wmz>  Typ@Intensität (stationär)
                gt_nonstat_<wmz>     Anomalietyp (nicht-stationär) oder NA
                gt_nonstat_label_<wmz>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stage4_features import build_rolling_features, ROLLING_LONG_H
from stage6_normalize import apply_scaler
from anomaly_injection import (
    STATIONARY_TYPES, NONSTATIONARY_TYPES,
    plan_events, compute_delta, compute_trend_delta,
    kind_series, label_series,
)
from models.registry import WMZ_NAMES


def _read(parquet_dir: Path, name: str) -> pd.DataFrame:
    """Parquet-Datei aus dem Ausgabeverzeichnis lesen, mit klarer Fehlermeldung.

    Fehlt die Datei, wird der Lauf mit dem Pfad abgebrochen statt mit einem
    nackten FileNotFoundError - das ist beim Nachvollziehen der
    Stage-Reihenfolge die nuetzlichere Auskunft.
    """
    path = parquet_dir / name
    if not path.is_file():
        raise SystemExit(f"Eingabe fehlt: {path}")
    return pd.read_parquet(path)


def build_injected(parquet_dir: Path,
                   scaler_dir: Path,
                   split: pd.Series,
                   target_split: str,
                   seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Siehe Modul-Docstring. Injiziert in alle Zeitstempel, deren
    Split-Zuordnung ``target_split`` ist (z. B. "val" oder "test")."""
    raw = _read(parquet_dir, "stage4_features_raw.parquet")
    res = _read(parquet_dir, "stage4_features_residual.parquet")
    stl = _read(parquet_dir, "stage3_stl.parquet")

    # Region-Maske ueber den (gemeinsamen) Stage-4-Index. reindex stellt
    # sicher, dass die Reihenfolge exakt zu raw passt.
    region = split.reindex(raw.index).eq(target_split).to_numpy()

    scaler_raw = pd.read_parquet(scaler_dir / "scaler_raw.parquet")
    scaler_res = pd.read_parquet(scaler_dir / "scaler_residual.parquet")
    scaler_trend = pd.read_parquet(scaler_dir / "scaler_trend.parquet")

    raw_inj = raw.copy()
    res_inj = res.copy()
    trend_inj = stl[[f"{w}_trend" for w in WMZ_NAMES]].copy()
    gt = pd.DataFrame(index=raw.index)

    # Ein gemeinsamer RNG sorgt fuer reproduzierbare, aber pro Seed
    # unterschiedliche Platzierung (Val-Seed != Test-Seed -> kein Tuning
    # auf die Test-Anomalien).
    rng = np.random.default_rng(seed)

    for wmz in WMZ_NAMES:
        # --- Variante A: stationäre Stoerung ins kW-Signal -----------------
        kw = raw[f"{wmz}_kw_mean"].to_numpy(dtype=float)
        stat_events = plan_events(kw, region, STATIONARY_TYPES, rng)
        delta = compute_delta(kw, stat_events)
        kw_inj = kw + delta
        raw_inj[f"{wmz}_kw_mean"] = kw_inj
        # Abgeleitete Rolling-Features auf dem gestoerten kW neu berechnen.
        rolling = build_rolling_features(pd.Series(kw_inj, index=raw.index), wmz)
        for c in rolling.columns:
            raw_inj[c] = rolling[c]

        # --- Variante B: dieselbe additive Differenz aufs Residuum ---------
        resid = res[f"{wmz}_residual"]
        resid_inj = resid + delta            # delta positionsgleich (gleicher Index)
        res_inj[f"{wmz}_residual"] = resid_inj
        res_inj[f"{wmz}_residual_rolling_std_{ROLLING_LONG_H}h"] = (
            resid_inj.rolling(ROLLING_LONG_H, min_periods=ROLLING_LONG_H // 2).std()
        )

        # --- Variante C: nicht-stationäre Stoerung in den Trend ------------
        tr = stl[f"{wmz}_trend"].to_numpy(dtype=float)
        ns_events = plan_events(tr, region, NONSTATIONARY_TYPES, rng)
        trend_inj[f"{wmz}_trend"] = tr + compute_trend_delta(tr, ns_events)

        # --- Ground-Truth (getrennt nach Schiene) --------------------------
        gt[f"gt_stat_{wmz}"] = kind_series(raw.index, stat_events)
        gt[f"gt_stat_label_{wmz}"] = label_series(raw.index, stat_events)
        gt[f"gt_nonstat_{wmz}"] = kind_series(raw.index, ns_events)
        gt[f"gt_nonstat_label_{wmz}"] = label_series(raw.index, ns_events)

    # Normalisieren mit den auf Train gefitteten Scalern (transform).
    frames = {
        "raw": apply_scaler(raw_inj, list(scaler_raw.index), scaler_raw),
        "residual": apply_scaler(res_inj, list(scaler_res.index), scaler_res),
        "trend": apply_scaler(trend_inj, list(scaler_trend.index), scaler_trend),
    }
    return frames, gt
