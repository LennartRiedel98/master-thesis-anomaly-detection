"""Generiert das Fall-Protokoll der qualitativen Evaluierung (Methodik 10b).

Schreibt ``docs/qualitative_evaluierung.md`` — das systematische Protokoll aus
Phase 8 des qualitativen Leitfadens. Es besteht aus datengetriebenen Teilen
(automatisch) und einer Interpretationsspalte (vom Autor manuell zu fuellen —
die fachliche Erklaerung ist bewusst *nicht* automatisiert, sie ist der
eigentliche wissenschaftliche Beitrag).

Aufbau des erzeugten Dokuments:

  A. Aggregierte Detektions-Matrix — Recall je Anomalietyp x Detektor,
     direkt aus ``stage10_metrics.csv`` (enthaelt alle Modelle inkl.
     LSTM-AE, exakt die berichteten Zahlen — kein Re-Fit).

  B. Event-Detektions-Matrix je (Variante x WMZ) — eine Zeile pro injiziertem
     Event, je Detektor ein Urteil ([ok] = TP erkannt / [x] = FN verpasst) plus
     der Spitzen-Score im Event relativ zur Schwelle (zeigt "knapp daneben").
     Hierfuer werden die klassischen Detektoren Stage-10-identisch neu
     gefittet (schnell); der LSTM-AE nur mit ``--include-lstm`` (langsam auf
     CPU). Die LSTM-Aggregat-Recalls stehen unabhaengig davon in Teil A.

  C. False-Positive-Kandidaten je Detektor — die staerksten Alarm-Segmente
     *ausserhalb* jedes GT-Events, mit Plot-Befehl zur Inspektion. Wichtig
     fuer die Frage, ob ein "Fehlalarm" ein echtes (Makro-)Ereignis ist.

  D. Empfohlene Faelle fuer die schriftliche Diskussion — kuratierte Auswahl
     (Showcase-TP + lehrreicher FN je Typ) mit fertigem Plot-Befehl und
     leeren Spalten Beobachtung/Erklaerung zum Ausfuellen.

Aufruf:
    python src/tools/qualitative_protocol.py                  # ohne LSTM
    python src/tools/qualitative_protocol.py --include-lstm   # mit LSTM (langsam)

Die Detektor-Logik (Fit, Schwelle, Event-Urteil) wird aus
``plot_qualitative_case.py`` importiert, damit Protokoll und Einzel-Plots
garantiert dieselben Zahlen verwenden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# src/ und src/tools/ auf den Pfad (wie in den anderen tools-Skripten).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import segments, threshold_from_quantile  # noqa: E402
from models.registry import REGISTRY, WMZ_NAMES, model_features  # noqa: E402
from stage7_train import select_training_rows  # noqa: E402
# Geprüfte Helfer aus dem Plot-Skript wiederverwenden (gleiche Zahlen).
from plot_qualitative_case import (  # noqa: E402
    extract_events, event_detected, stat_kind, resolve_model_cls,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "gebaeude_a"

# Reihenfolge der Anomalietypen (stationaer zuerst), wie in der Injektionskarte.
TYPE_ORDER = ["spike", "drop", "plateau", "leakage", "drift", "structural_break"]
MODEL_ORDER = ["zscore", "lof", "iforest", "constancy", "pelt", "lstm_ae"]


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--include-lstm", action="store_true",
                   help="LSTM-AE in die Event-Matrix aufnehmen (Re-Fit, "
                        "langsam auf CPU). Aggregat-Recall steht ohnehin in Teil A.")
    p.add_argument("--device", default="cpu",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--threshold-quantile", type=float, default=0.99)
    p.add_argument("--top-fp", type=int, default=5,
                   help="Anzahl staerkster False-Positive-Segmente je Detektor.")
    p.add_argument("--out", default="docs/qualitative_evaluierung.md")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Detektor fitten + scoren (Stage-10-identisch, via plot_qualitative_case).
# --------------------------------------------------------------------------- #
def fit_score(dataset, variant, wmz, model_name, clean, inj, split,
              device, q) -> tuple[pd.Series, float]:
    """Reproduziert Stage 10: Fit auf sauberem Train, Schwelle aus Val-Quantil,
    Score auf injiziertem Test. Gibt (scores, threshold) zurueck."""
    import json
    model_cls = resolve_model_cls(variant, model_name)
    best_path = ROOT / "outputs" / dataset / "hpo" / "best_hparams.json"
    best = json.loads(best_path.read_text()) if best_path.is_file() else {}
    hp = (best.get(variant, {}).get(wmz, {})
          .get(model_name, {}).get("hparams", {})) or {}
    if device != "auto" and model_name == "lstm_ae":
        hp = {**hp, "device": device}

    X_clean = model_features(clean, variant, wmz)
    train_idx = select_training_rows(clean, wmz, variant)
    model = model_cls(**hp).fit(X_clean.loc[train_idx])

    if model_name == "pelt":
        threshold = 0.5
    else:
        val_index = split.index[split == "val"]
        threshold = threshold_from_quantile(model.score(X_clean.loc[val_index]), q)
    return model.score(inj), threshold


def event_peak(scores: pd.Series, start, end) -> float:
    """Maximaler (nicht-NaN) Score innerhalb eines Event-Fensters."""
    seg = scores.loc[start:end].to_numpy(dtype=float)
    seg = seg[~np.isnan(seg)]
    return float(seg.max()) if seg.size else float("nan")


def fp_segments(scores: pd.Series, threshold: float, test_index: pd.Index,
                event_mask: np.ndarray, top: int) -> list[dict]:
    """Staerkste Alarm-Segmente *ausserhalb* jedes GT-Events (FP-Kandidaten)."""
    s = scores.reindex(test_index).to_numpy(dtype=float)
    alarm = np.where(np.isnan(s), False, s > threshold) & ~event_mask
    out = []
    for a, b in segments(alarm):
        peak = float(np.nanmax(s[a:b]))
        out.append({"start": test_index[a], "end": test_index[b - 1],
                    "hours": b - a, "peak": peak})
    out.sort(key=lambda d: d["peak"], reverse=True)
    return out[:top]


# --------------------------------------------------------------------------- #
# Markdown-Bausteine.
# --------------------------------------------------------------------------- #
def section_a(metrics: pd.DataFrame) -> list[str]:
    """Aggregierte Recall-Matrix je Anomalietyp × Detektor (ueber WMZ summiert)."""
    L = ["## A. Aggregierte Detektions-Matrix (Recall je Typ × Detektor)", "",
         "Quelle: `stage10_metrics.csv` (Point-adjusted Recall, über alle drei "
         "WMZ summiert: erkannte / injizierte Events). Enthält **alle** Modelle "
         "inkl. LSTM-AE — exakt die im Ergebnisbericht berichteten Zahlen.", ""]
    typ = metrics[metrics["stratum"].str.startswith("type:")].copy()
    typ["type"] = typ["stratum"].str.split(":").str[1]
    # Summiere detected/n_true je (model, type) ueber Varianten & WMZ.
    agg = (typ.groupby(["model", "type"])[["n_pred", "n_true"]].sum()
              .reset_index())
    types = [t for t in TYPE_ORDER if t in agg["type"].unique()]
    models = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    L.append("| Detektor | " + " | ".join(types) + " |")
    L.append("|" + "---|" * (len(types) + 1))
    for m in models:
        cells = []
        for t in types:
            row = agg[(agg["model"] == m) & (agg["type"] == t)]
            if row.empty:
                cells.append("–")
            else:
                d, n = int(row["n_pred"].iloc[0]), int(row["n_true"].iloc[0])
                rec = d / n if n else float("nan")
                cells.append(f"{d}/{n} ({rec:.0%})")
        L.append(f"| **{m}** | " + " | ".join(cells) + " |")
    L += ["", "*Lesart:* `erkannt/injiziert (Recall)`. „–\" = Detektor läuft "
          "nicht auf der Schiene dieses Typs (z. B. PELT nur auf Trend → nur "
          "drift/structural_break; Constancy nur auf raw).", "",
          "*Hinweis zu den Nennern:* zscore/lof/iforest laufen auf **zwei** "
          "Schienen (raw **und** residual), ihre injizierten Events zählen "
          "daher doppelt (z. B. spike: 2 Varianten × 3 WMZ); constancy (nur "
          "raw) und pelt (nur trend) haben entsprechend kleinere Nenner. Der "
          "Recall ist je Detektor korrekt, die absoluten Event-Zahlen sind "
          "zwischen den Detektoren also **nicht** 1:1 vergleichbar — die "
          "schienenscharfe Aufschlüsselung liefert Teil B.", ""]
    return L


def section_coverage(per_wmz: dict, models_run: list[str]) -> list[str]:
    """Abdeckungsanalyse: Union-Recall je Typ (≥1 Detektor erkennt das Event).

    Beantwortet die Kernfrage der Detektor-Suite: Wird jeder Anomalietyp von
    *mindestens einem* Detektor abgedeckt? Die stationären GT-Labels sind
    variantenunabhängig (raw und residual teilen ``gt_stat``), ein physisches
    Event wird also von allen Detektoren beider Schienen beurteilt; wir
    vereinigen (logisches ODER) über alle anwendbaren (Variante × Modell).
    """
    L = ["## A2. Abdeckungsanalyse — Union-Recall der Detektor-Suite", "",
         "Pro injiziertem Event: erkennt es **mindestens ein** Detektor "
         "(logisches ODER über alle Detektoren der zuständigen Schiene)? Zeigt, "
         "ob die komplementäre Suite gemeinsam jeden Typ abdeckt — die "
         "eigentliche Aussage hinter dem Mehr-Detektoren-Ansatz. Vergleich: "
         "bestes Einzelmodell vs. Union.", "",
         "| Typ | Events | Bestes Einzelmodell | Union (≥1 Detektor) | Zugewinn |",
         "|---|---|---|---|---|"]
    # Pro (Schiene, WMZ) die Event-Liste + alle Verdict-Arrays sammeln. Schiene
    # = stat (raw/residual) bzw. nonstat (trend); raw und residual teilen die
    # identische gt_stat-Event-Liste, daher per Schiene zusammenfassbar.
    schiene_of = {"raw": "stat", "residual": "stat", "trend": "nonstat"}
    bucket: dict = {}   # (schiene, wmz) -> {"events":[...], "verdicts":{model:[...]}}
    for (variant, wmz), data in per_wmz.items():
        key = (schiene_of[variant], wmz)
        b = bucket.setdefault(key, {"events": data["events"], "verdicts": {}})
        for m, pm in data["per_model"].items():
            # Mehrere Varianten je Schiene (raw+residual): ODER-verknüpfen.
            tag = f"{variant}:{m}"
            b["verdicts"][tag] = pm["verdict"]

    # Aggregation je Anomalietyp über alle WMZ.
    per_type: dict = {}   # type -> {"total":n, "union":n, "best":{model:hits}}
    for (schiene, wmz), b in bucket.items():
        events = b["events"]
        for i, ev in enumerate(events):
            t = ev["type"]
            agg = per_type.setdefault(t, {"total": 0, "union": 0, "best": {}})
            agg["total"] += 1
            hit_any = any(v[i] for v in b["verdicts"].values())
            agg["union"] += int(hit_any)
            for tag, v in b["verdicts"].items():
                agg["best"][tag] = agg["best"].get(tag, 0) + int(v[i])

    for t in TYPE_ORDER:
        if t not in per_type:
            continue
        agg = per_type[t]
        tot = agg["total"]
        union_r = agg["union"] / tot if tot else float("nan")
        best_tag, best_hits = max(agg["best"].items(), key=lambda kv: kv[1])
        best_r = best_hits / tot if tot else float("nan")
        best_model = best_tag.split(":")[1]
        gain = union_r - best_r
        L.append(f"| {t} | {tot} | {best_model}: {best_hits}/{tot} ({best_r:.0%}) "
                 f"| {agg['union']}/{tot} (**{union_r:.0%}**) | +{gain:.0%} |")
    lstm_note = ("" if "lstm_ae" in models_run else
                 " **Achtung:** ohne `--include-lstm` zählt der LSTM-AE hier "
                 "nicht zur Union — besonders relevant für drift/"
                 "structural_break, wo er der einzige Ko-Detektor neben PELT "
                 "ist; die Trend-Union ist dann unterschätzt.")
    L += ["", "*Lesart:* „Union\" = Anteil Events, die **irgendein** Detektor "
          "der Schiene fängt. Ein großer Zugewinn gegenüber dem besten "
          "Einzelmodell belegt, dass sich die Detektoren ergänzen (kein "
          "einzelnes Modell deckt alle Typen ab). Hinweis: Die Union maximiert "
          "den Recall, erhöht aber auch die Fehlalarmrate (Union der "
          "False Positives) — die Präzisionsseite ist gesondert zu bewerten."
          + lstm_note, ""]
    return L


def section_b(per_wmz: dict, models_run: list[str]) -> list[str]:
    """Event-Detektions-Matrix je (Variante × WMZ): Zeile = Event, Spalte = Detektor."""
    L = ["## B. Event-Detektions-Matrix je WMZ", "",
         "Eine Zeile pro injiziertem Event. ✓ = als TP erkannt (irgendein Punkt "
         "im Event über Schwelle, point-adjusted), ✗ = verpasst (FN). Der Wert "
         "in Klammern ist der **Spitzen-Score im Event ÷ Schwelle** — Werte knapp "
         "unter 1.00 markieren „fast erkannt\".", ""]
    for (variant, wmz), data in per_wmz.items():
        events = data["events"]
        if not events:
            continue
        cols = [m for m in MODEL_ORDER if m in models_run and m in data["per_model"]]
        L.append(f"### {variant} / {wmz}  (Schiene: gt_{stat_kind(variant)})")
        L.append("")
        L.append("| # | Event | Start | Dauer | " + " | ".join(cols) + " |")
        L.append("|---|---|---|---|" + "---|" * len(cols))
        for i, ev in enumerate(events, 1):
            dur = int((ev["end"] - ev["start"]) / pd.Timedelta(hours=1)) + 1
            name = ev["type"] + (f"@{ev['intensity']}" if ev["intensity"] else "")
            cells = []
            for m in cols:
                pm = data["per_model"][m]
                det = pm["verdict"][i - 1]
                ratio = pm["ratio"][i - 1]
                mark = "✓" if det else "✗"
                rstr = f" ({ratio:.2f})" if np.isfinite(ratio) else ""
                cells.append(f"{mark}{rstr}")
            L.append(f"| {i} | {name} | {ev['start']:%Y-%m-%d %H:%M} | "
                     f"{dur} h | " + " | ".join(cells) + " |")
        L.append("")
    return L


def section_c(fp: dict, dataset: str) -> list[str]:
    """False-Positive-Kandidaten je Detektor + Plot-Befehl zur Inspektion."""
    L = ["## C. False-Positive-Kandidaten (Alarme außerhalb jedes GT-Events)", "",
         "Die stärksten Alarm-Segmente, die **kein** injiziertes Event treffen. "
         "Zentrale Frage der qualitativen Analyse: echter Fehlalarm oder reale "
         "(Makro-)Anomalie? Mit dem Plot-Befehl das Fenster inspizieren und gegen "
         "`monatlich_makro_events.png` (COVID/EnSikuMaV) abgleichen.", ""]
    for (variant, wmz, model), segs in fp.items():
        if not segs:
            continue
        L.append(f"### {variant} / {wmz} / {model}")
        L.append("")
        L.append("| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |")
        L.append("|---|---|---|---|---|")
        for s in segs:
            pad = "--start %s --end %s" % (
                (s["start"] - pd.Timedelta(days=3)).date(),
                (s["end"] + pd.Timedelta(days=3)).date())
            cmd = (f"`python src/tools/plot_qualitative_case.py --variant "
                   f"{variant} --wmz {wmz} --model {model} {pad}`")
            L.append(f"| {s['start']:%Y-%m-%d %H:%M} | {s['end']:%Y-%m-%d %H:%M} "
                     f"| {s['hours']} h | {s['peak']:.3f} | {cmd} |")
        L.append("")
    return L


def parse_preserved(out_path: Path) -> dict[str, tuple[str, str]]:
    """Bereits ausgefüllte Beobachtung/Erklärung aus einem früheren Teil D lesen.

    So überleben manuell (oder KI-gestützt) verfasste Interpretationen ein
    erneutes Generieren — nur die faktischen Spalten werden aktualisiert.
    Schlüssel = Fall-ID (F01, F02, …).
    """
    preserved: dict[str, tuple[str, str]] = {}
    if not out_path.is_file():
        return preserved
    for line in out_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("| F"):
            continue
        cells = [c.strip() for c in s.split("|")]
        # ['', 'F01 (...)', variant, typ, start, urteil, cmd, beob, erkl, '']
        if len(cells) < 9:
            continue
        fid = cells[1].split()[0]   # "F01"
        beob, erkl = cells[-3], cells[-2]
        if beob or erkl:
            preserved[fid] = (beob, erkl)
    return preserved


def section_d(per_wmz: dict, preserved: dict[str, tuple[str, str]]) -> list[str]:
    """Kuratierte Fälle (Showcase-TP + lehrreicher FN je Typ).

    Bereits ausgefüllte Interpretationsspalten (``preserved``, nach Fall-ID)
    werden übernommen; neue Fälle bleiben leer.
    """
    L = ["## D. Empfohlene Fälle für die schriftliche Diskussion", "",
         "Kuratierte Auswahl: je Anomalietyp ein klarer Treffer (Showcase) und "
         "ein lehrreicher Fehlschlag. Spalten **Beobachtung** und **Erklärung / "
         "Theorie-Bezug** sind die fachliche Deutung; bereits ausgefüllte Zellen "
         "bleiben beim Neugenerieren erhalten (Merge über die Fall-ID).", "",
         "| Fall | Variante/WMZ/Detektor | Typ | Event-Start | Urteil | "
         "Plot-Befehl | Beobachtung | Erklärung / Theorie-Bezug |",
         "|---|---|---|---|---|---|---|---|"]
    fid = 0
    # Pro Typ: bester TP-Fall (höchstes ratio bei Erkennung) + lehrreichster FN
    # (höchstes ratio bei Nicht-Erkennung = "knapp verpasst").
    by_type: dict[str, list] = {t: [] for t in TYPE_ORDER}
    for (variant, wmz), data in per_wmz.items():
        for m, pm in data["per_model"].items():
            for i, ev in enumerate(data["events"]):
                by_type.setdefault(ev["type"], []).append(
                    {"variant": variant, "wmz": wmz, "model": m, "ev": ev,
                     "det": pm["verdict"][i], "ratio": pm["ratio"][i]})
    for t in TYPE_ORDER:
        cand = by_type.get(t, [])
        if not cand:
            continue
        tps = [c for c in cand if c["det"]]
        fns = [c for c in cand if not c["det"]]
        picks = []
        if tps:
            picks.append(("Showcase-TP", max(tps, key=lambda c: c["ratio"])))
        if fns:
            picks.append(("Lehr-FN", max(fns, key=lambda c: c["ratio"])))
        for tag, c in picks:
            fid += 1
            ev = c["ev"]
            it = ev["intensity"]
            # Event eindeutig ueber den Start-Zeitstempel adressieren (mehrere
            # Events desselben typ@intensitaet existieren).
            cmd = (f"`python src/tools/plot_qualitative_case.py --variant "
                   f"{c['variant']} --wmz {c['wmz']} --model {c['model']} "
                   f"--event-start '{ev['start']:%Y-%m-%d %H:%M}'`")
            verdict = "TP" if c["det"] else "FN"
            name = ev["type"] + (f"@{it}" if it else "")
            beob, erkl = preserved.get(f"F{fid:02d}", ("", ""))
            L.append(f"| F{fid:02d} ({tag}) | {c['variant']}/{c['wmz']}/"
                     f"{c['model']} | {name} | {ev['start']:%Y-%m-%d %H:%M} | "
                     f"{verdict} | {cmd} | {beob} | {erkl} |")
    L.append("")
    return L


def main() -> None:
    """Das Fall-Protokoll der qualitativen Evaluierung erzeugen.

    Schreibt ``docs/qualitative_evaluierung.md``: Recall-Matrix nach Typ und
    Detektor, Ereignis-Matrix je Zaehler, Fehlalarm-Kandidaten und kuratierte
    Faelle. Die Deutungsspalte bleibt bewusst leer - die fachliche Erklaerung
    ist nicht automatisierbar und der eigentliche wissenschaftliche Beitrag.
    """
    args = parse_args()
    parquet = ROOT / "outputs" / args.dataset / "parquet"
    reports = ROOT / "outputs" / args.dataset / "reports"

    metrics = pd.read_csv(reports / "stage10_metrics.csv")
    split = pd.read_parquet(parquet / "split_assignment.parquet")["split"]
    gt = pd.read_parquet(parquet / "stage9_ground_truth.parquet")
    test_index = split.index[split == "test"]

    clean_cache: dict[str, pd.DataFrame] = {}
    inj_cache: dict[str, pd.DataFrame] = {}

    def clean_of(v):
        """Unveraenderte Merkmalstabelle einer Schiene liefern (beim ersten Zugriff geladen)."""
        if v not in clean_cache:
            clean_cache[v] = pd.read_parquet(parquet / f"stage6_normalized_{v}.parquet")
        return clean_cache[v]

    def inj_of(v):
        """Injizierte Merkmalstabelle einer Schiene liefern (beim ersten Zugriff geladen)."""
        if v not in inj_cache:
            inj_cache[v] = pd.read_parquet(parquet / f"stage9_injected_{v}.parquet")
        return inj_cache[v]

    skip = set() if args.include_lstm else {"lstm_ae"}
    models_run: list[str] = []
    per_wmz: dict = {}
    fp: dict = {}

    print("Detektor-Re-Fits für die Event-Matrix (Teil B/C)…")
    for variant, model_classes in REGISTRY.items():
        names = [c.name for c in model_classes if c.name not in skip]
        for wmz in WMZ_NAMES:
            label_col = f"gt_{stat_kind(variant)}_label_{wmz}"
            events = extract_events(gt.loc[test_index, label_col])
            # Boolean-Maske der Event-Stunden (fuer FP-Abgrenzung).
            ev_mask = gt.loc[test_index, f"gt_{stat_kind(variant)}_{wmz}"].notna().to_numpy()
            per_model: dict = {}
            for model_name in names:
                scores, thr = fit_score(args.dataset, variant, wmz, model_name,
                                        clean_of(variant), inj_of(variant), split,
                                        args.device, args.threshold_quantile)
                verdict = [event_detected(scores, thr, e["start"], e["end"])
                           for e in events]
                ratio = [event_peak(scores, e["start"], e["end"]) / thr
                         if thr else float("nan") for e in events]
                per_model[model_name] = {"verdict": verdict, "ratio": ratio}
                fp[(variant, wmz, model_name)] = fp_segments(
                    scores, thr, test_index, ev_mask, args.top_fp)
                if model_name not in models_run:
                    models_run.append(model_name)
                print(f"  {variant}/{wmz}/{model_name}: "
                      f"{sum(verdict)}/{len(events)} TP")
            per_wmz[(variant, wmz)] = {"events": events, "per_model": per_model}

    # --- Dokument zusammensetzen ------------------------------------------- #
    lines = [
        "# Qualitative Evaluierung — Fall-Protokoll", "",
        "*Automatisch erzeugt von "
        "[src/tools/qualitative_protocol.py](../src/tools/qualitative_protocol.py).*",
        f"Datensatz: `{args.dataset}`. Schwelle: "
        f"{args.threshold_quantile:.0%}-Quantil der sauberen Validierungs-Scores "
        "(PELT binär bei 0.5), Stage-10-identisch. Einzelfälle visuell: "
        "`plot_qualitative_case.py` (siehe Plot-Befehle unten).", "",
        "Die faktischen Teile (A, A2, B, C — Recall/Urteil/Score/FP) sind "
        "datengetrieben; "
        "die **Interpretation** (Teil D, Spalten Beobachtung/Erklärung) ist "
        "bewusst manuell auszufüllen — sie ist der eigentliche wissenschaftliche "
        "Beitrag, nicht automatisierbar ohne Fabrikation.", "",
        "---", "",
    ]
    lines += section_a(metrics)
    lines += ["---", ""]
    lines += section_coverage(per_wmz, models_run)
    lines += ["---", ""]
    lines += section_b(per_wmz, models_run)
    lines += ["---", ""]
    lines += section_c(fp, args.dataset)
    lines += ["---", ""]
    # Vorhandene Interpretationen (Teil D) erhalten, falls schon ausgefüllt.
    out_path = ROOT / args.out
    lines += section_d(per_wmz, parse_preserved(out_path))

    # Zielordner anlegen: Der Default liegt in docs/, das im Abgabe-Repo
    # nicht mehr existiert - ohne mkdir liefe das Skript dort ins Leere.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    note = "" if args.include_lstm else "  (ohne LSTM-AE — siehe Teil A für dessen Recall)"
    print(f"\n-> {out_path}{note}")


if __name__ == "__main__":
    main()
