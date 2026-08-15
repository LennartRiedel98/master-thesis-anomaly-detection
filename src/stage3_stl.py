"""Stage 3 - MSTL-Zerlegung (Trend + Tages- + Wochensaison + Residuum).

Zerlegt jede der drei stuendlichen kW-Reihen mit Multi-Seasonal-Trend
Decomposition (MSTL) in den Perioden 24 h (Tagesgang) und 168 h
(Wochenrhythmus).

Verwendung in den Folge-Stages:
- ``wmz_*_trend``, ``wmz_*_seasonal_24h``, ``wmz_*_seasonal_168h``
  -> beschreibende Musteranalyse (Verschiebung des Grundniveaus ueber die
     Jahre, typische Tages- und Wochenprofile)
- ``wmz_*_residual``
  -> Eingangssignal der Schiene B fuer die Detektoren (Z-Score, LOF,
     Isolation Forest, LSTM-AE)

Entwurfsentscheidungen:
- MSTL mit den Perioden [24, 168] und ``robust=True``: Der LOESS-Fit
  gewichtet Ausreisser im Residuum iterativ ab, damit einzelne Anomalien
  Trend und Saison nicht zu sich heranziehen koennen.
- Ein einziger Fit ueber die gesamte Vier-Jahres-Reihe. STL glaettet lokal,
  der Informationsfluss ueber die Train/Val/Test-Grenzen hinweg ist deshalb
  gering; dieser Rest-Nachteil wird zugunsten von Einfachheit und einer
  durchgehenden, sprungfreien Zerlegung in Kauf genommen.
- Die STL-Implementierung von statsmodels vertraegt keine NaN. Fehlende
  Werte werden **nur fuer den Fit** linear interpoliert; im ``residual``
  wird an genau diesen Stellen anschliessend wieder NaN gesetzt, damit kein
  Modell auf erfundenen Werten trainiert. Trend und Saison behalten dort
  ihre gefitteten Werte - als glatte Fortschreibung sind sie fuer die
  Musteranalyse brauchbar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "demo_synthetic"

WMZ_KW_COLS = ["wmz_1_kw_mean", "wmz_2_kw_mean", "wmz_3_kw_mean"]
PERIODS = [24, 168]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen (nur die Datensatz-Wahl)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Name des Unterordners unter outputs/. "
                        f"Standard: {DEFAULT_DATASET}")
    return p.parse_args()


def decompose_one(series: pd.Series, periods: list[int]) -> pd.DataFrame:
    """Eine Reihe zerlegen; liefert Trend, beide Saisonanteile und Residuum.

    Das Residuum ist an allen Stellen NaN, an denen schon die Eingangsreihe
    NaN war. Damit trainiert kein nachgelagertes Modell auf den zuvor
    interpolierten Stuetzwerten - die Interpolation dient ausschliesslich
    dazu, den STL-Fit ueberhaupt rechenbar zu machen.
    """
    was_nan = series.isna()
    filled = series.interpolate(method="linear", limit_direction="both")

    res = MSTL(filled, periods=periods,
               stl_kwargs={"robust": True}).fit()

    out = pd.DataFrame(index=series.index)
    out["trend"] = res.trend
    for col in res.seasonal.columns:           # 'seasonal_24', 'seasonal_168'
        out[col + "h"] = res.seasonal[col]     # -> 'seasonal_24h' usw.
    out["residual"] = res.resid
    out.loc[was_nan, "residual"] = np.nan
    return out


def variance_share(series: pd.Series, component: pd.Series) -> float:
    """Varianzanteil einer Komponente an der Eingangsreihe, auf [0, 1] begrenzt.

    Nur als Plausibilitaetskontrolle im Qualitaetsbericht gedacht: Die
    Anteile summieren sich wegen der Kovarianz zwischen den Komponenten
    nicht auf 1. Die belastbare Kennzahl ist die Strength-Definition nach
    Wang, Smith und Hyndman - die rechnet ``src/exploration/stl_strength.py``.
    """
    valid = series.notna() & component.notna()
    total_var = series.loc[valid].var()
    if total_var == 0:
        return 0.0
    return float(np.clip(component.loc[valid].var() / total_var, 0.0, 1.0))


def main() -> None:
    """Stage 3 ausfuehren: Stage-2-Reihen zerlegen und Ergebnis schreiben.

    Zerlegt die drei kW-Reihen nacheinander, haengt die Komponenten
    spaltenweise zusammen und gibt einen Qualitaetsbericht aus (Varianz-
    anteile je Komponente, Residuen-Statistik). Ausgabe:
    ``outputs/<dataset>/parquet/stage3_stl.parquet``.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    in_path = out_root / "parquet" / "stage2_hourly.parquet"
    out_path = out_root / "parquet" / "stage3_stl.parquet"

    if not in_path.is_file():
        raise SystemExit(
            f"Stage-2-Ausgabe nicht gefunden: {in_path}\n"
            f"Vorher ausfuehren: python src/stage2_preprocess.py "
            f"--dataset {args.dataset}"
        )

    print(f"Datensatz: {args.dataset}")
    print(f"Lade {in_path}")
    df = pd.read_parquet(in_path)
    print(f"  Form: {df.shape}, verwendete Perioden: {PERIODS}")

    parts = []
    for col in WMZ_KW_COLS:
        wmz = col.replace("_kw_mean", "")
        print(f"  Zerlege {col} ...", flush=True)
        comp = decompose_one(df[col], PERIODS)
        comp.columns = [f"{wmz}_{c}" for c in comp.columns]
        parts.append(comp)

    out = pd.concat(parts, axis=1)

    # Qualitaetsbericht -----------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 3 (MSTL) - Qualitaetsbericht")
    print("=" * 70)
    print(f"  Form der Ausgabe: {out.shape}")
    print(f"  Spalten         : {list(out.columns)}")

    print("\n  Varianzanteil je Komponente (grobe Plausibilitaetskontrolle):")
    print(f"    {'Zaehler':<8} {'Trend':>8} {'Saison_24h':>12} "
          f"{'Saison_168h':>12} {'Residuum':>10}")
    for col in WMZ_KW_COLS:
        wmz = col.replace("_kw_mean", "")
        s = df[col]
        v_t = variance_share(s, out[f"{wmz}_trend"])
        v_d = variance_share(s, out[f"{wmz}_seasonal_24h"])
        v_w = variance_share(s, out[f"{wmz}_seasonal_168h"])
        v_r = variance_share(s, out[f"{wmz}_residual"])
        print(f"    {wmz:<8} {v_t:>8.3f} {v_d:>12.3f} "
              f"{v_w:>12.3f} {v_r:>10.3f}")

    print("\n  Residuen-Statistik (NaN an den urspruenglichen Luecken):")
    for col in WMZ_KW_COLS:
        wmz = col.replace("_kw_mean", "")
        r = out[f"{wmz}_residual"].dropna()
        print(f"    {wmz}_residual  Mittel={r.mean():>+8.2f}  "
              f"Std={r.std():>7.2f}  "
              f"Min={r.min():>+9.1f}  Max={r.max():>+9.1f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"\n  Geschrieben: {out_path}")


if __name__ == "__main__":
    main()
