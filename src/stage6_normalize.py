"""Stage 6 - Normalisierung (StandardScaler pro Variante).

Erzeugt fuer jede Detektions-Variante (A/raw, B/residual, C/trend) eine
normalisierte Feature-Tabelle. Der Scaler wird ausschliesslich auf
Train-Zeilen gefittet (Stage-5-Assignment) und dann auf Train/Val/Test
angewendet. Mean und Std werden persistiert, damit spaetere Inferenz
auf neuen Daten dieselbe Skalierung verwendet.

Spec (Methodik 7.3):
    StandardScaler (pro Track separat), fit ausschliesslich auf
    Training-Daten, transform auf Val und Test. Scaler werden
    persistiert fuer spaetere Inferenz.

Inputs:
    outputs/<ds>/parquet/stage4_features_raw.parquet       (Variante A)
    outputs/<ds>/parquet/stage4_features_residual.parquet  (Variante B)
    outputs/<ds>/parquet/stage3_stl.parquet                (Variante C, nur trend)
    outputs/<ds>/parquet/split_assignment.parquet

Outputs:
    outputs/<ds>/parquet/stage6_normalized_<variant>.parquet
    outputs/<ds>/scalers/scaler_<variant>.parquet

Designentscheidungen:
- NaN-sicher via pandas (mean/std skipna, Arithmetik propagiert NaN).
  Damit muessen wir an Luecken-Positionen keinen Platzhalter erfinden;
  spaetere Modelle muessen NaN-Zeilen explizit ausschliessen.
- Population-Std (ddof=0) zur Konsistenz mit sklearn.StandardScaler.
- Meta-Spalten (was_flagged, interpolated) werden 1:1 durchgereicht,
  nicht normalisiert - sie sind KEIN Modell-Input, sondern dienen der
  Trainings-Daten-Filterung in Stage 7.
- Die Spalte ``split`` aus Stage 5 wird ebenfalls durchgereicht, damit
  Stage 7 nur eine Datei laden muss.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "gebaeude_a"

WMZ_NAMES = ["wmz_1", "wmz_2", "wmz_3"]


@dataclass(frozen=True)
class VariantSpec:
    """Beschreibt eine Detektions-Variante fuer Stage 6.

    feature_file:  Parquet, das die Input-Spalten enthaelt
    select_cols:   Funktion, die aus diesem Parquet die zu skalierenden
                   Spalten zurueckgibt (Reihenfolge ist stabil)

    Wir halten die drei Varianten als Datenobjekte, damit die Hauptschleife
    nur ueber ``VARIANTS`` iteriert. Neue Varianten lassen sich dadurch
    durch Hinzufuegen eines Eintrags ergaenzen, ohne dass die Logik
    aenderten werden muss (Open/Closed-Prinzip).
    """
    name: str
    feature_file: str
    select_cols: callable   # (df) -> list[str]


def _raw_feature_cols(df: pd.DataFrame) -> list[str]:
    """Variante A: alles ausser Meta-Spalten.

    Meta-Spalten (was_flagged, interpolated) sind boolean und werden zur
    Trainings-Daten-Filterung in Stage 7 verwendet, aber NICHT als
    Modell-Input - sie wuerden in einem Produktiv-Setting zur Inferenzzeit
    gar nicht bekannt sein.
    """
    return [c for c in df.columns
            if not (c.endswith("_was_flagged") or c.endswith("_interpolated"))]


def _residual_feature_cols(df: pd.DataFrame) -> list[str]:
    """Variante B: alles ausser Meta-Spalten (gleiche Logik wie raw)."""
    return [c for c in df.columns
            if not (c.endswith("_was_flagged") or c.endswith("_interpolated"))]


def _trend_feature_cols(df: pd.DataFrame) -> list[str]:
    """Variante C: ausschliesslich die drei Trend-Spalten aus Stage 3.

    Stage 3 schreibt 4 Spalten pro Zaehler (trend, seasonal_24h,
    seasonal_168h, residual). Variante C nutzt aber NUR den Trend - dort
    sind nicht-stationaere Anomalien (Drift, Strukturbruch) sichtbar.
    Saisonalitaeten und Residuum kommen in Variante B zum Einsatz.
    """
    return [f"{wmz}_trend" for wmz in WMZ_NAMES]


VARIANTS: list[VariantSpec] = [
    VariantSpec("raw",      "stage4_features_raw.parquet",      _raw_feature_cols),
    VariantSpec("residual", "stage4_features_residual.parquet", _residual_feature_cols),
    VariantSpec("trend",    "stage3_stl.parquet",               _trend_feature_cols),
]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Name des Unterordners unter outputs/. "
                        f"Default: {DEFAULT_DATASET}")
    return p.parse_args()


def fit_scaler(df: pd.DataFrame,
               cols: list[str],
               train_mask: pd.Series) -> pd.DataFrame:
    """Berechnet mean und std (ddof=0) pro Spalte auf Train-Zeilen.

    NaN-Werte werden ignoriert (pandas skipna=True). Spalten mit
    std=0 (z. B. konstante Flags falls versehentlich included) werden
    auf std=1.0 gesetzt, damit nachfolgende Division nicht zu inf/NaN
    fuehrt. Solche Spalten haben dann transformierte Werte = 0.
    """
    # WICHTIG: ausschliesslich Train-Zeilen sehen -> kein Leakage von
    # Val/Test in die Skalierungs-Statistik.
    sub = df.loc[train_mask, cols]

    # skipna=True ist Default, hier explizit zur Dokumentation:
    # Lueckenpositionen aus Stage 2/3 haetten sonst alle Spalten-
    # statistiken auf NaN gezogen.
    mean = sub.mean(skipna=True)

    # ddof=0 (Population-Std) zur Kompatibilitaet mit sklearn.StandardScaler.
    # Der Unterschied zu ddof=1 ist bei n=25k Trainstunden vernachlaessigbar
    # (Faktor sqrt(N/(N-1)) ~ 1 + 2e-5), aber wir wollen exakte Reproduzier-
    # barkeit zu jeder Standardbibliothek.
    std = sub.std(skipna=True, ddof=0)

    # Konstante Spalten: std=0 -> Division durch Null. .where(cond, other)
    # ersetzt Werte WO cond False ist, hier also wo std nicht > 0.
    std = std.where(std > 0, 1.0)

    # DataFrame mit Index=Spaltennamen und zwei Wertespalten ist die
    # serialisierbare Form des Scalers - kompatibel mit pd.read_parquet.
    return pd.DataFrame({"mean": mean, "std": std})


def apply_scaler(df: pd.DataFrame,
                 cols: list[str],
                 scaler: pd.DataFrame) -> pd.DataFrame:
    """(value - mean) / std. NaN bleibt NaN.

    pandas-Broadcasting macht das implizit: ``scaler["mean"]`` ist eine
    Series indiziert nach Spaltennamen, ``df[cols]`` ist ein DataFrame
    mit identischen Spalten - Subtraktion alignt automatisch ueber den
    Spaltennamen.
    """
    return (df[cols] - scaler["mean"]) / scaler["std"]


def process_variant(variant: VariantSpec,
                    split: pd.Series,
                    parquet_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Liest die Feature-Datei, skaliert, gibt (normalized_df, scaler_df) zurueck."""
    src_path = parquet_dir / variant.feature_file
    if not src_path.is_file():
        raise SystemExit(f"  Input fehlt fuer Variante {variant.name!r}: {src_path}")

    df = pd.read_parquet(src_path)

    # Welche Spalten werden tatsaechlich skaliert?
    feature_cols = variant.select_cols(df)
    # Rest = alles andere = Meta-Spalten, die wir 1:1 durchreichen.
    meta_cols = [c for c in df.columns if c not in feature_cols]

    # Index ausrichten (alle drei Files teilen denselben hourly index).
    # Defensive: falls jemand Stage 3/4 mit veraendertem Clipping laufen
    # liesse, fangen wir die Diskrepanz hier statt mit kryptischem
    # Boolean-Indexing-Error weiter unten.
    if not df.index.equals(split.index):
        df = df.reindex(split.index)

    # Train-Maske = Boolean-Serie mit True dort, wo split == "train".
    train_mask = (split == "train")
    scaler = fit_scaler(df, feature_cols, train_mask)
    scaled = apply_scaler(df, feature_cols, scaler)

    # Zusammensetzen: skalierte Features + Meta + Split-Spalte. Die
    # Split-Spalte direkt hier mit reinzuschreiben spart Stage 7 ein
    # zusaetzliches Join.
    out = pd.concat([scaled, df[meta_cols], split.to_frame("split")], axis=1)
    out.index.name = "timestamp"
    return out, scaler


def main() -> None:
    """Stage 6 ausfuehren: je Schiene einen Scaler fitten und anwenden.

    Der Scaler wird ausschliesslich auf den Train-Zeilen gefittet und dann
    auf Train, Validierung und Test angewandt. Wuerde er auf allen Daten
    gefittet, floesse die Verteilung des Testzeitraums in die Skalierung ein -
    ein subtiler, aber echter Informationsabfluss. Mittelwert und Streuung
    werden mitgeschrieben, damit spaetere Inferenz dieselbe Skala verwendet.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    scaler_dir = out_root / "scalers"

    split_path = parquet_dir / "split_assignment.parquet"
    if not split_path.is_file():
        raise SystemExit(
            f"Split nicht gefunden: {split_path}\n"
            f"Run: python src/stage5_split.py --dataset {args.dataset}"
        )

    print(f"Dataset: {args.dataset}")
    split = pd.read_parquet(split_path)["split"]
    print(f"  Split geladen: {len(split)} Zeitstempel "
          f"(train={(split == 'train').sum()}, "
          f"val={(split == 'val').sum()}, "
          f"test={(split == 'test').sum()})")

    scaler_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Stage 6 Quality Report")
    print("=" * 70)

    for variant in VARIANTS:
        print(f"\n  Variante {variant.name!r}")
        out, scaler = process_variant(variant, split, parquet_dir)

        n_feat = len(scaler)
        # NaN-Anteil ueber alle Feature-Zellen (mittlere Zeile x mittlere Spalte)
        # gibt eine kompakte Lueckenquote, die wir mit Stage 2/3 vergleichen
        # koennen (sollte naeherungsweise gleich sein).
        nan_share = out[scaler.index].isna().mean().mean() * 100
        print(f"    Feature-Spalten skaliert: {n_feat}")
        print(f"    Output-Shape            : {out.shape}")
        print(f"    Mittlerer NaN-Anteil (Features): {nan_share:5.2f} %")
        # Sanity: nach Skalierung sollten Train-Spalten mean~0, std~1 haben.
        # Wenn etwas hier deutlich groesser als Maschinen-Epsilon ist, ist
        # entweder die Skalierung kaputt oder der Train-Slice falsch.
        train_slice = out.loc[split == "train", scaler.index]
        mean_check = train_slice.mean(skipna=True).abs().max()
        std_check = (train_slice.std(skipna=True, ddof=0) - 1.0).abs().max()
        print(f"    Train-Sanity            : max|mean|={mean_check:.2e}, "
              f"max|std-1|={std_check:.2e}")

        out_path = parquet_dir / f"stage6_normalized_{variant.name}.parquet"
        scaler_path = scaler_dir / f"scaler_{variant.name}.parquet"
        out.to_parquet(out_path)
        scaler.to_parquet(scaler_path)
        print(f"    Wrote {out_path}")
        print(f"    Wrote {scaler_path}")


if __name__ == "__main__":
    main()
