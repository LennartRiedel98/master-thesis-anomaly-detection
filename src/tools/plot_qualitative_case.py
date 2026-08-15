"""Qualitative Fallanalyse — Overlay-Plot je (Variante x WMZ x Detektor x Event).

Das visuelle Pendant zur quantitativen Stage-10-Auswertung: statt einer
aggregierten Metrik zeigt dieses Werkzeug *einen konkreten Fall* und macht
sichtbar, ob ein Detektor zeitlich und formal richtig reagiert. Pro Aufruf
entsteht eine Abbildung mit drei uebereinandergelegten Ebenen (Methodik 10b):

    1. Rohsignal im kW-Massstab (interpretierbarste Ebene, aus
       stage9_injected_raw) im gewaehlten Zeitfenster.
    2. Ground-Truth-Band(er): das/die injizierte(n) Event(s) als farbiges
       axvspan aus stage9_ground_truth — die Wahrheit.
    3. Detektor-Score auf zweiter y-Achse + Schwellwertlinie; Stunden mit
       Score > Schwelle (= Alarm) sind rot schraffiert.

So ist auf einen Blick ablesbar: deckt sich der Alarm mit dem Band (True
Positive), liegt das Band ohne Alarm da (False Negative) oder feuert der
Alarm ausserhalb (False Positive)?

Das Skript reproduziert die Stage-10-Logik exakt (gleiche HPs aus
best_hparams.json, gleicher Schwellwert = q-Quantil der sauberen
Validierungs-Scores, PELT binaer mit Schwelle 0.5), damit die hier
gezeigten Faelle 1:1 zu den berichteten Metriken passen.

Aufruf:
    # 1) Erst die Events einer (Variante,WMZ) auflisten + sehen, was der
    #    Detektor je Event erkennt (Fallauswahl, Phase 1 des Guides):
    python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 \\
        --model iforest --list

    # 2) Dann einen konkreten Fall plotten (Event automatisch lokalisiert):
    python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 \\
        --model iforest --type spike --intensity 10

    # 3) Oder ein freies Zeitfenster (z. B. fuer False-Positive-Analyse):
    python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 \\
        --model constancy --start 2023-07-01 --end 2023-07-20

LSTM-AE: auf dem Mac immer ``--device cpu`` (der MPS-nn.LSTM-Kernel ist
defekt). Default ist daher 'cpu', nicht 'auto'.

Figuren landen in ``outputs/<ds>/figures/qualitativ/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src/ auf den Pfad legen, damit die Pipeline-Module (wie in Stage 10)
# als Top-Level importierbar sind — identisch zu src/tools/data_sweep.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import threshold_from_quantile  # noqa: E402
from models.registry import REGISTRY, WMZ_NAMES, model_features  # noqa: E402
from stage7_train import select_training_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # src/tools/ -> Projekt-Wurzel
DEFAULT_DATASET = "demo_synthetic"

# Konsistente Farben je Anomalietyp (identisch zu plot_injektionskarte.py).
TYPE_COLOR = {
    "spike": "#d62728",
    "drop": "#1f77b4",
    "plateau": "#9467bd",
    "leakage": "#ff7f0e",
    "drift": "#2ca02c",
    "structural_break": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen.

    ``--list`` zeigt alle injizierten Ereignisse samt Treffer-Urteil, ohne zu
    zeichnen - damit laesst sich ein interessanter Fall auswaehlen, bevor man
    den teuren Plot rechnet.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--variant", required=True, choices=list(REGISTRY))
    p.add_argument("--wmz", required=True, choices=WMZ_NAMES)
    p.add_argument("--model", required=True,
                   help="Modellname, z. B. iforest, lof, zscore, pelt, "
                        "constancy, lstm_ae (muss in der Variante existieren).")
    p.add_argument("--type", default=None,
                   help="Anomalietyp zum automatischen Lokalisieren des "
                        "Fensters (spike, drop, plateau, leakage, drift, "
                        "structural_break).")
    p.add_argument("--intensity", default=None,
                   help="Intensitaet zur Eingrenzung (z. B. 10 fuer spike@10). "
                        "Ohne Angabe: erstes Event des Typs.")
    p.add_argument("--event-start", default=None,
                   help="Exakter Start-Zeitstempel (ISO) eines Events, um es "
                        "eindeutig zu adressieren (noetig, wenn mehrere Events "
                        "desselben typ@intensitaet existieren). Hat Vorrang vor "
                        "--type/--intensity.")
    p.add_argument("--start", default=None, help="Fenster-Start (ISO), "
                   "Alternative zu --type fuer freie Fenster (FP-Analyse).")
    p.add_argument("--end", default=None, help="Fenster-Ende (ISO).")
    p.add_argument("--pad-days", type=float, default=5.0,
                   help="Tage Rand links/rechts um das lokalisierte Event.")
    p.add_argument("--threshold-quantile", type=float, default=0.99)
    p.add_argument("--device", default="cpu",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device fuer den LSTM-AE-Re-Fit. Default 'cpu' "
                        "(Mac: MPS-nn.LSTM defekt).")
    p.add_argument("--list", action="store_true",
                   help="Nur die Events der (Variante,WMZ) auflisten (inkl. "
                        "ob der Detektor sie erkennt) und beenden.")
    return p.parse_args()


def resolve_model_cls(variant: str, model_name: str):
    """Modell-Klasse aus der Registry holen — mit klarer Fehlermeldung."""
    for cls in REGISTRY[variant]:
        if cls.name == model_name:
            return cls
    available = ", ".join(c.name for c in REGISTRY[variant])
    raise SystemExit(f"Modell {model_name!r} nicht in Variante {variant!r}. "
                     f"Verfuegbar: {available}")


def stat_kind(variant: str) -> str:
    """Welche GT-Schiene gilt fuer die Variante (Methodik 6.2 / 10a)?"""
    return "nonstat" if variant == "trend" else "stat"


def extract_events(label_series: pd.Series) -> list[dict]:
    """Zusammenhaengende Label-Laeufe als Event-Liste (typ, intensitaet, span).

    Ein Event = maximaler Lauf gleicher Label-Werte. Das Label hat die Form
    ``typ@intensitaet`` (z. B. ``spike@10``) oder nur ``typ`` (z. B.
    ``plateau``). NA = kein Event.
    """
    s = label_series.fillna("")
    idx = s.index
    events: list[dict] = []
    prev, start = "", None
    for i, v in enumerate(s.to_numpy()):
        if v != prev:
            if prev != "" and start is not None:
                events.append(_make_event(prev, idx[start], idx[i - 1]))
            start = i if v != "" else None
            prev = v
    if prev != "" and start is not None:
        events.append(_make_event(prev, idx[start], idx[-1]))
    return events


def _make_event(label: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Ein Ground-Truth-Label in ein Ereignis-Woerterbuch zerlegen.

    Die Labels haben die Form ``typ@intensitaet`` (etwa ``spike@10``) oder
    nur ``typ``, wenn der Anomalietyp keine Intensitaetsstufen kennt.
    """
    kind, _, intensity = label.partition("@")
    return {"label": label, "type": kind, "intensity": intensity,
            "start": start, "end": end}


def fit_and_score(args, clean, inj, split):
    """Stage-10-identisch: Modell fitten, Schwelle bestimmen, Test scoren.

    Rueckgabe: (scores ueber den vollen injizierten Index, threshold).
    """
    model_cls = resolve_model_cls(args.variant, args.model)

    best_path = ROOT / "outputs" / args.dataset / "hpo" / "best_hparams.json"
    best = json.loads(best_path.read_text()) if best_path.is_file() else {}
    hp = (best.get(args.variant, {}).get(args.wmz, {})
          .get(args.model, {}).get("hparams", {})) or {}
    # Device nur dem LSTM-AE mitgeben (klassische Modelle kennen ihn nicht).
    if args.device != "auto" and args.model == "lstm_ae":
        hp = {**hp, "device": args.device}

    X_clean = model_features(clean, args.variant, args.wmz)
    train_idx = select_training_rows(clean, args.wmz, args.variant)
    model = model_cls(**hp).fit(X_clean.loc[train_idx])

    # Schwellwert: PELT ist binaer (0/1) -> 0.5; sonst q-Quantil der
    # sauberen Validierungs-Scores (unabhaengig vom Test-Set).
    if args.model == "pelt":
        threshold = 0.5
    else:
        val_index = split.index[split == "val"]
        val_scores = model.score(X_clean.loc[val_index])
        threshold = threshold_from_quantile(val_scores, args.threshold_quantile)

    # Auf der vollen injizierten Reihe scoren (LSTM braucht den Kontext).
    scores = model.score(inj)
    return scores, threshold


def event_detected(scores: pd.Series, threshold: float,
                   start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Point-adjusted: Event erkannt, wenn *irgendein* Punkt darin alarmiert."""
    seg = scores.loc[start:end].to_numpy(dtype=float)
    if seg.size == 0:
        return False
    return bool(np.any(np.where(np.isnan(seg), False, seg > threshold)))


def list_events(args, events, scores, threshold) -> None:
    """Alle Events der (Variante,WMZ) tabellarisch + Detektor-Urteil ausgeben."""
    print(f"\nEvents fuer {args.variant}/{args.wmz} "
          f"(Schiene: gt_{stat_kind(args.variant)}), Modell {args.model}:")
    print(f"  Schwellwert = {threshold:.4f}\n")
    print(f"  {'Typ':<18}{'Intens.':<9}{'Start':<20}{'Dauer':>7}  Urteil")
    print("  " + "-" * 64)
    for ev in events:
        dur_h = int((ev['end'] - ev['start']) / pd.Timedelta(hours=1)) + 1
        det = event_detected(scores, threshold, ev['start'], ev['end'])
        verdict = "TP (erkannt)" if det else "FN (verpasst)"
        print(f"  {ev['type']:<18}{ev['intensity'] or '-':<9}"
              f"{str(ev['start']):<20}{dur_h:>5}h  {verdict}")
    print()


def select_window(args, events) -> tuple[pd.Timestamp, pd.Timestamp, dict | None]:
    """Zeitfenster bestimmen: frei (--start/--end), exaktes Event
    (--event-start) oder erstes Event eines Typs (--type)."""
    if args.start and args.end:
        return pd.Timestamp(args.start), pd.Timestamp(args.end), None

    pad = pd.Timedelta(days=args.pad_days)

    # Exaktes Event ueber seinen Start-Zeitstempel adressieren - noetig, wenn
    # mehrere Events desselben typ@intensitaet existieren (z. B. ein TP- und
    # ein FN-Fall von structural_break@0.2).
    if args.event_start:
        target = pd.Timestamp(args.event_start)
        match = [e for e in events if e["start"] == target]
        if not match:
            raise SystemExit(f"Kein Event mit Start {target} in "
                             f"{args.variant}/{args.wmz}. --list zeigt alle Starts.")
        ev = match[0]
        return ev["start"] - pad, ev["end"] + pad, ev

    if not args.type:
        raise SystemExit("Bitte --type, --event-start oder --start/--end "
                         "angeben. Tipp: --list zeigt alle verfuegbaren Events.")

    candidates = [e for e in events if e["type"] == args.type
                  and (args.intensity is None or e["intensity"] == args.intensity)]
    if not candidates:
        raise SystemExit(f"Kein Event vom Typ {args.type!r}"
                         + (f"@{args.intensity}" if args.intensity else "")
                         + f" in {args.variant}/{args.wmz}. --list zeigt, was da ist.")
    ev = candidates[0]
    return ev["start"] - pad, ev["end"] + pad, ev


def plot_case(args, raw_signal, signal_label, scores, threshold, events,
              win_start, win_end, focus) -> Path:
    """Den Overlay-Plot (kW + GT-Band + Score + Schwelle) zeichnen + speichern."""
    sig = raw_signal.loc[win_start:win_end]
    sc = scores.loc[win_start:win_end]

    fig, ax_kw = plt.subplots(figsize=(14, 5.5))

    # Ebene 1: Signal in kW (linke Achse) - roh bzw. Trend je nach Variante.
    ax_kw.plot(sig.index, sig.to_numpy(), color="#333333", lw=1.0,
               label=signal_label)
    ax_kw.set_ylabel(f"{args.wmz}  Leistung [kW]")
    ax_kw.set_xlabel("Zeit")
    ax_kw.grid(True, alpha=0.25)
    # x-Achse fest aufs Fenster begrenzen, damit ein laenger als das Fenster
    # reichendes GT-Band (z. B. mehrwoechiger Drift) die Achse nicht dehnt.
    ax_kw.set_xlim(win_start, win_end)

    # Ebene 2: Ground-Truth-Baender aller Events im Fenster (ans Fenster
    # geklemmt, damit ueberstehende Events die Achse nicht aufziehen).
    seen_types: set[str] = set()
    for ev in events:
        if ev["end"] < win_start or ev["start"] > win_end:
            continue
        col = TYPE_COLOR.get(ev["type"], "#999999")
        lbl = f"GT: {ev['type']}" if ev["type"] not in seen_types else None
        seen_types.add(ev["type"])
        ax_kw.axvspan(max(ev["start"], win_start), min(ev["end"], win_end),
                      color=col, alpha=0.20, label=lbl)

    # Ebene 3: Score + Schwelle (rechte Achse) + Alarm-Schraffur.
    ax_sc = ax_kw.twinx()
    ax_sc.plot(sc.index, sc.to_numpy(), color="#1f77b4", lw=1.1,
               label="Detektor-Score")
    ax_sc.axhline(threshold, color="#d62728", lw=1.0, linestyle="--",
                  label=f"Schwelle = {threshold:.3f}")
    ax_sc.set_ylabel("Score")
    # Alarm-Stunden (Score > Schwelle) leicht rot hinterlegen.
    alarm = sc[sc > threshold]
    for ts in alarm.index:
        ax_sc.axvspan(ts, ts + pd.Timedelta(hours=1), color="#d62728",
                      alpha=0.10, lw=0)

    # Urteil fuer das fokussierte Event (falls eines lokalisiert wurde).
    verdict = ""
    if focus is not None:
        det = event_detected(scores, threshold, focus["start"], focus["end"])
        verdict = (f"  —  {focus['type']}"
                   + (f"@{focus['intensity']}" if focus["intensity"] else "")
                   + (": TP (erkannt)" if det else ": FN (verpasst)"))

    ax_kw.set_title(f"Qualitativer Fall: {args.variant}/{args.wmz}/{args.model}"
                    f"{verdict}", fontsize=12)
    ax_kw.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))

    # Legenden beider Achsen zusammenfuehren.
    h1, l1 = ax_kw.get_legend_handles_labels()
    h2, l2 = ax_sc.get_legend_handles_labels()
    ax_kw.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()

    fig_dir = ROOT / "outputs" / args.dataset / "figures" / "qualitativ"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Dateiname eindeutig: bei exakt adressiertem Event den Event-Start mit
    # aufnehmen (sonst kollidieren TP- und FN-Fall desselben typ@intensitaet).
    if focus is not None:
        it = f"_{focus['intensity']}" if focus["intensity"] else ""
        tag = f"{focus['type']}{it}_{focus['start']:%Y%m%d%H}"
    else:
        tag = f"{win_start:%Y%m%d}_{win_end:%Y%m%d}"
    out = fig_dir / f"{args.variant}_{args.wmz}_{args.model}_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Overlay-Plot fuer einen konkreten Fall erzeugen.

    Drei uebereinandergelegte Ebenen: das Rohsignal in kW als
    interpretierbarste Ebene, das Ground-Truth-Band des injizierten
    Ereignisses und der Detektor-Score mit seiner Schwelle. Das visuelle
    Gegenstueck zur aggregierten Metrik - es zeigt, ob ein Detektor zeitlich
    und in der Form richtig reagiert, nicht nur ob eine Kennzahl stimmt.
    """
    args = parse_args()
    parquet = ROOT / "outputs" / args.dataset / "parquet"

    split = pd.read_parquet(parquet / "split_assignment.parquet")["split"]
    gt = pd.read_parquet(parquet / "stage9_ground_truth.parquet")
    clean = pd.read_parquet(parquet / f"stage6_normalized_{args.variant}.parquet")
    inj = pd.read_parquet(parquet / f"stage9_injected_{args.variant}.parquet")
    # Rohsignal in kW immer aus der raw-Injektion (interpretierbarste Ebene),
    # unabhaengig von der Variante des Detektors.
    # Anzuzeigendes Signal je Variante so waehlen, dass es (a) das ist, was
    # der Detektor sieht, und (b) die injizierte Anomalie enthaelt:
    #   raw/residual -> rohes kW-Signal (stationaere Anomalien liegen hier),
    #   trend        -> MSTL-Trend (nicht-stationaere Anomalien liegen hier;
    #                   im rohen kW-Signal waere ein Drift kaum sichtbar).
    # Beide werden mit dem zugehoerigen Scaler (mean/std je Spalte, fit auf
    # Train) in physikalische kW zurueckgerechnet: kW = x*std + mean.
    if args.variant == "trend":
        sig_frame = inj
        sig_col = f"{args.wmz}_trend"
        scaler = pd.read_parquet(parquet.parent / "scalers" / "scaler_trend.parquet")
        signal_label = "Trend [kW] (injiziert)"
    else:
        sig_frame = pd.read_parquet(parquet / "stage9_injected_raw.parquet")
        sig_col = f"{args.wmz}_kw_mean"
        scaler = pd.read_parquet(parquet.parent / "scalers" / "scaler_raw.parquet")
        signal_label = "Signal [kW] (injiziert)"
    raw_signal = (sig_frame[sig_col] * scaler.loc[sig_col, "std"]
                  + scaler.loc[sig_col, "mean"])

    # Events nur im Test-Fenster (dort wird injiziert).
    test_index = split.index[split == "test"]
    label_col = f"gt_{stat_kind(args.variant)}_label_{args.wmz}"
    events = extract_events(gt.loc[test_index, label_col])

    scores, threshold = fit_and_score(args, clean, inj, split)

    if args.list:
        list_events(args, events, scores, threshold)
        return

    win_start, win_end, focus = select_window(args, events)
    out = plot_case(args, raw_signal, signal_label, scores, threshold, events,
                    win_start, win_end, focus)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
