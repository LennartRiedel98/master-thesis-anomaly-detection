"""Stage 7 - Modelltraining (Orchestrierung).

Iteriert ueber alle (Variante x WMZ x Modell)-Jobs aus
``models.registry`` und ruft fit() auf den Train-Zeilen auf. Alle sechs
Detektoren sind implementiert (siehe ``src/models/*.py``); die
Orchestrierung ist von der Modell-Implementierung getrennt, sodass ein
weiterer Detektor nur einen Registry-Eintrag braucht und die
Pipeline-Struktur unberuehrt bleibt.

Trainings-Daten-Filterung (Methodik 6.4 / 7.4):
    Stunden mit aktiven Fehler-Flags des betroffenen WMZ
    (``<wmz>_was_flagged == True``) werden vor fit() entfernt. Damit
    lernen die Modelle keine Datenfehler als Normalmuster. Die Flags
    selbst sind kein Modell-Input.

Kleinere Datensaetze / Probelaeufe:
    ``--max-train-rows N`` begrenzt das Training auf die juengsten N
    Trainings-Stunden (zusammenhaengender Tail, LSTM-Fenster-vertraeglich).
    Nuetzlich fuer schnelle Iteration oder Tests auf schwacher Hardware.

Persistenz:
    outputs/<ds>/models/<variant>/<wmz>/<model_name>.pkl

``NotImplementedError`` wird abgefangen und als uebersprungener Job
geloggt statt den Lauf abzubrechen - relevant nur, wenn ein neu
registrierter Detektor noch keine fit()-Implementierung hat. Bei den
sechs Detektoren der Arbeit tritt der Fall nicht ein. Andere Exceptions
propagieren.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.registry import REGISTRY, WMZ_NAMES, iter_training_jobs, model_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "demo_synthetic"


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente dieser Stage einlesen.

    ``--variants``, ``--wmz`` und ``--models`` schraenken den Lauf auf eine
    Teilmenge ein - noetig, um die teuren LSTM-Jobs getrennt von den
    klassischen Detektoren zu rechnen.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Name des Unterordners unter outputs/. "
                        f"Default: {DEFAULT_DATASET}")
    p.add_argument("--variants", nargs="+", default=None,
                   choices=list(REGISTRY),
                   help="Nur diese Varianten trainieren. Default: alle.")
    p.add_argument("--wmz", nargs="+", default=None, choices=WMZ_NAMES,
                   help="Nur diese WMZ trainieren. Default: alle.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Nur diese Modellnamen (z. B. zscore lof iforest "
                        "pelt). Default: alle. Nuetzlich, um z. B. lstm_ae "
                        "auf Maschinen ohne GPU/torch auszulassen.")
    p.add_argument("--max-train-rows", type=int, default=None, metavar="N",
                   help="Auf hoechstens N Trainings-Stunden begrenzen "
                        "(juengster zusammenhaengender Zeitausschnitt). "
                        "Gedacht fuer schnelle Probelaeufe / kleinere "
                        "Datensaetze; Default: alle Trainings-Stunden.")
    return p.parse_args()


def load_variant_frame(parquet_dir: Path, variant: str) -> pd.DataFrame:
    """Laedt das Stage-6-Output fuer eine Variante."""
    path = parquet_dir / f"stage6_normalized_{variant}.parquet"
    if not path.is_file():
        raise SystemExit(
            f"Stage-6 Output fehlt fuer {variant!r}: {path}\n"
            f"Run: python src/stage6_normalize.py"
        )
    return pd.read_parquet(path)


def select_training_rows(df: pd.DataFrame,
                         wmz: str,
                         variant: str,
                         max_rows: int | None = None) -> pd.Index:
    """Train-Zeilen ohne aktive Fehler-Flags fuer diesen WMZ.

    Fuer die Trend-Variante (Variante C / PELT) ist die Filterung
    konzeptionell weniger sinnvoll - PELT laeuft auf dem geglaetteten
    Trend, in dem Punkt-Fehler schon weitgehend absorbiert sind.
    Wir filtern trotzdem konsistent, damit alle Varianten dasselbe
    Trainings-Fenster sehen.

    Konkret: ein Modell fuer wmz_2 sieht keine Stunden, in denen
    ``wmz_2_was_flagged == True`` ist (siehe Methodik 6.4). Die anderen
    WMZ-Flags sind irrelevant, weil das Modell deren Signale ohnehin
    nicht sieht (model_features filtert sie raus).

    ``max_rows`` begrenzt optional auf die juengsten N Trainings-Stunden.
    Wir nehmen bewusst einen zusammenhaengenden Tail-Ausschnitt (kein
    Zufalls-Subsampling), weil der LSTM-Autoencoder gleitende Fenster
    ueber aufeinanderfolgende Stunden bildet - eine zufaellige Stichprobe
    wuerde die Sequenz-Struktur zerstoeren. Der juengste Ausschnitt liegt
    zudem zeitlich am naechsten an Val/Test.
    """
    # Boolean-Maske: True = Zeile darf zum Training verwendet werden.
    train = df["split"] == "train"
    flag_col = f"{wmz}_was_flagged"
    if flag_col in df.columns:
        # ~ = logisches NICHT. fillna(False) wegen NaN-Stunden in der
        # Flag-Spalte (Stage 2 setzt NaN, wenn dort gar keine Daten waren -
        # die schliessen wir besser nicht aus, weil die NaN-Filterung
        # spaeter ueber die Feature-Spalten passiert).
        train &= ~df[flag_col].fillna(False).astype(bool)
    idx = df.index[train]
    # Tail-Begrenzung: df ist nach Zeitstempel sortiert, also sind die
    # letzten N Eintraege der juengste zusammenhaengende Block.
    if max_rows is not None and len(idx) > max_rows:
        idx = idx[-max_rows:]
    return idx


def main() -> None:
    """Stage 7 ausfuehren: alle ausgewaehlten (Schiene x Zaehler x Modell)-Jobs."""
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    parquet_dir = out_root / "parquet"
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    if args.max_train_rows is not None:
        print(f"  Trainings-Limit: max. {args.max_train_rows} Stunden pro Job "
              f"(juengster Tail) - Probelauf-Modus")

    # Variant-Frames lazy laden - sparen Memory, falls nur eine Teilmenge
    # ueber --variants gewuenscht ist. Cache verhindert Mehrfach-Read fuer
    # dieselbe Variante (passiert sonst 4x fuer raw, 4x fuer residual,
    # 2x fuer trend).
    cache: dict[str, pd.DataFrame] = {}

    def frame(variant: str) -> pd.DataFrame:
        """Merkmalstabelle einer Schiene liefern, beim ersten Zugriff geladen.

        Der Cache spart das mehrfache Einlesen derselben Parquet-Datei: Pro
        Schiene greifen bis zu fuenf Modelle x drei Zaehler auf dieselbe Tabelle zu.
        """
        if variant not in cache:
            cache[variant] = load_variant_frame(parquet_dir, variant)
        return cache[variant]

    # CLI-Filter (--variants / --wmz / --models). Wenn nicht gesetzt,
    # default = alle. Set fuer O(1)-Membership-Check in der Hauptschleife.
    selected_variants = set(args.variants) if args.variants else set(REGISTRY)
    selected_wmz = set(args.wmz) if args.wmz else set(WMZ_NAMES)
    # --models filtert auf den Klassen-Attribut-Namen jeder Detektor-Klasse
    # (z. B. "zscore", "lof", "iforest", "pelt", "lstm_ae"). None = alle.
    selected_models = set(args.models) if args.models else None

    print("\n" + "=" * 70)
    print("Stage 7 - Trainings-Jobs")
    print("=" * 70)

    n_done, n_stub, n_skip = 0, 0, 0
    # iter_training_jobs() yieldet (variant, wmz, model_cls) in stabiler
    # Reihenfolge -> deterministische Logs ueber Laeufe hinweg.
    for variant, wmz, model_cls in iter_training_jobs():
        # CLI-Filter anwenden, bevor wir das DataFrame anfassen.
        if (variant not in selected_variants
                or wmz not in selected_wmz
                or (selected_models is not None
                    and model_cls.name not in selected_models)):
            n_skip += 1
            continue

        df = frame(variant)
        # Modell-Input vorbereiten: Slicing nach Variante x WMZ +
        # Trainings-Daten-Filterung.
        X = model_features(df, variant, wmz)
        train_idx = select_training_rows(df, wmz, variant, args.max_train_rows)
        X_train = X.loc[train_idx]

        # Modell mit Default-HPs instanziieren. Spaeter ersetzt Stage 8
        # die Default-HPs durch optimierte (Grid- oder Random-Search).
        model = model_cls()
        tag = f"{variant}/{wmz}/{model.name}"
        print(f"  {tag:<32}  X_train={X_train.shape}  ", end="")

        try:
            model.fit(X_train)
        except NotImplementedError as e:
            # Detektor ohne fit()-Implementierung: kein Abbruchgrund, sondern
            # ein uebersprungener Job. Tritt bei den sechs Detektoren der
            # Arbeit nicht auf - nur relevant, wenn jemand einen neuen
            # Detektor registriert, bevor er ihn ausimplementiert.
            print(f"[stub] {e}")
            n_stub += 1
            continue

        # Persistenz-Pfad: outputs/<ds>/models/<variant>/<wmz>/<name>.pkl
        # Diese Hierarchie macht es leicht, Modelle pro Variante und WMZ
        # separat zu inspizieren oder zu loeschen.
        target = models_dir / variant / wmz / f"{model.name}.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
        # Device-Tag nur fuer Modelle, die einen GPU/CPU-Kontext halten
        # (LSTM-AE). Klassische Detektoren rechnen ohnehin auf CPU; ein
        # leerer Tag haelt den Log kompakt. So sieht man im Konsolen-
        # Output direkt, dass das Training tatsaechlich auf cuda lief
        # (Windows-Task-Manager zeigt LSTM-Compute nur unter dem "Cuda"-
        # bzw. "Compute_0"-Engine-Reiter, nicht im Standard-"3D"-Tab).
        dev_tag = f"[{model._device}] " if getattr(model, "_device", None) else ""
        print(f"fitted {dev_tag}-> {target.relative_to(out_root)}")
        n_done += 1

    # Schlussbilanz - ein vollstaendiger Lauf ergibt n_stub = 0 und
    # n_done = 33 (raw: 5 Modelle x 3 WMZ = 15, residual: 4 x 3 = 12,
    # trend: 2 x 3 = 6; der Constancy-Detektor laeuft nur auf raw).
    print("\n" + "-" * 70)
    print(f"  trainiert : {n_done}")
    print(f"  Stub      : {n_stub}  (NotImplementedError - implementieren in src/models/)")
    print(f"  uebersprungen (Flag): {n_skip}")


if __name__ == "__main__":
    main()
