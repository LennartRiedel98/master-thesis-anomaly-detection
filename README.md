# Mustererkennung und Anomaliedetektion in Heizenergieverbrauchsdaten

Begleitendes Code-Repository zur Masterarbeit **„Potenzialanalyse von KI für
Mustererkennung und Anomaliedetektion von Energieverbrauchsdaten"** —
untersucht an den **Wärmemengenzählern (WMZ)** eines Gewerbegebäudes
(HTW Berlin, SS 2026). Dieses Repository enthält den vollständigen Code und
die Anleitung, um **alle Ergebnisse und Abbildungen der Arbeit zu
reproduzieren**.

> **Diese Fassung enthält keine echten Messdaten.** Die Arbeit wurde auf
> vertraulichen Betriebsdaten gerechnet; die liegen hier nicht bei. Stattdessen
> erzeugt ein Generator einen **synthetischen Datensatz im identischen
> Dateiformat**, mit dem die gesamte Pipeline durchläuft — ein Befehl, siehe
> [Daten](#daten) und [Schritt 0](#schritt-0--demo-datensatz-erzeugen).

> **Worum geht es?** Heizenergie-Zeitreihen aus Gebäudezählern tragen
> ausgeprägte Muster — Tages- und Wochenrhythmus, Heizsaison, Abhängigkeit
> von der Außentemperatur — und darin Mess-, Anlagen- und
> Verbrauchsanomalien, die für ein effizientes Energiemanagement
> automatisiert erkannt werden sollen. Die Arbeit geht deshalb in zwei
> Schritten vor: **erst die Muster herausarbeiten, dann die Abweichungen
> davon suchen.** Für den zweiten Schritt vergleicht sie klassische und
> neuronale Verfahren auf einer einheitlichen, reproduzierbaren Pipeline
> und bewertet sie gegen kontrolliert injizierte Anomalien.

## Methodischer Überblick

### Teil 1 — Mustererkennung

Bevor eine Abweichung auffallen kann, muss das Erwartbare beschrieben sein.
Dieser Teil zerlegt die Zeitreihen und macht ihre Struktur messbar:

- **MSTL-Zerlegung** (`src/stage3_stl.py`) trennt jede Reihe in **Trend**,
  zwei **Saisonkomponenten** — `periods=[24, 168]`, also Tagesgang und
  Wochenrhythmus — und **Residuum**. Die beiden Perioden sind nicht geraten,
  sondern über ACF/PACF an den Daten belegt
  (`src/exploration/autocorrelation.py`).
- **Strukturstärke (STL-Strength)** nach Wang, Smith und Hyndman 2006
  beziffert je Komponente ihr Signal-zu-Rausch-Verhältnis gegenüber dem
  Residuum — kein Varianzanteil, die Werte konkurrieren also nicht
  miteinander (`src/exploration/stl_strength.py`).
- **Betriebsmuster und Kopplung:** Tages-/Wochenprofile, Jahresdauerlinie,
  Stunde-×-Wochentag-Heatmap und die zählerübergreifende Korrelation zeigen,
  dass die drei Zähler verschiedene Nutzungen abbilden
  (`src/tools/stage2_explore.py`, `stage3_explore.py`, `src/exploration/`).
- **Außentemperatur als Haupteinflussgröße:** LOWESS-Streudiagramm,
  Kreuzkorrelation, **Mutual Information** und eine Regime-Betrachtung an der
  Heizgrenze — die Beziehung ist nichtlinear, weshalb Pearson allein sie
  unterschätzt.
- **Makroereignisse:** COVID-Einbruch und die EnSikuMaV-Sparverordnung werden
  als reale Strukturbrüche sichtbar gemacht
  (`src/tools/stage3_macroevents.py`). Der EnSikuMaV-Stichtag dient später
  als **labelfreier Validierungsanker** — ein echtes Ereignis mit bekanntem
  Datum, gegen das sich die Changepoint-Detektion prüfen lässt, ohne dafür
  etwas injizieren zu müssen.

Diese Muster sind **nicht nur Vorarbeit, sondern tragen die Detektion**: Die
Schienen B und C unten sind genau die Residuum- und die Trendkomponente aus
der MSTL-Zerlegung.

### Teil 2 — Anomaliedetektion

Die Detektion erfolgt über **drei parallele Varianten** derselben
Zeitreihe, die unterschiedliche Anomalietypen sichtbar machen:

- **Variante A — Rohdaten (kW):** erkennt stationäre Anomalien direkt im
  Leistungssignal.
- **Variante B — MSTL-Residuum:** entfernt Saisonalität (Tages-/Wochen­
  rhythmus) und Trend; erkennt stationäre Anomalien gegen den erwarteten
  Verlauf.
- **Variante C — MSTL-Trend:** isoliert die langsame Komponente; erkennt
  **nicht-stationäre** Anomalien (Drift, strukturelle Brüche).

A/B und C sind dabei bewusst disjunkt zuständig. Es kommen **sechs
Detektoren** zum Einsatz:

| Detektor | Typ | Variante(n) | Quelle |
|---|---|---|---|
| Z-Score (saisonal) | statistisch | A, B | Hochenbaum u. a. 2017 |
| Local Outlier Factor (LOF) | dichtebasiert | A, B | Breunig u. a. 2000 |
| Isolation Forest | ensemble-basiert | A, B | Liu u. a. 2008 |
| Constancy / Plateau | varianz-change-point | A | Inclán & Tiao 1994; Killick u. a. 2012 |
| PELT | changepoint-detektion | C | Killick u. a. 2012 |
| LSTM-Autoencoder | neuronal (Rekonstruktion) | A, B, C | Malhotra u. a. 2016 |

Der **Constancy-Detektor** schließt eine Lücke der übrigen Verfahren: Plateaus
(eingefrorene Werte trotz erwarteter Aktivität) sind *kollektive* Anomalien, die
weder Punkt- noch Rekonstruktionsdetektoren erkennen.

**Bewertung:** Da reale Daten keine vollständigen Labels haben, werden
**synthetische Anomalien** kontrolliert injiziert (Spikes, Drops,
Plateaus, Leckagen, Drift, Strukturbrüche). Die Evaluation nutzt die
Point-Adjust-Metrik (Xu u. a. 2018) samt Kritik daran (Kim u. a. 2022)
und ist nach Anomalietyp und -intensität stratifiziert. Getrennte Seeds
für Validierung (HPO) und Test verhindern eine optimistische Verzerrung.

## Daten

Die Pipeline hat **genau eine Eingabe**: einen Ordner `data/<dataset>/` mit den
Logging-CSVs der Wärmemengenzähler und einer Open-Meteo-CSV. Alles Weitere wird
daraus erzeugt.

> ### ⚠️ Die Originaldaten sind nicht Teil dieser Veröffentlichung
>
> Ausgewertet wurden **reale Betriebsdaten** eines Gewerbegebäudes. Weil sie
> dem Betriebs- und Geschäftsinteresse des Wärmeversorgers unterliegen, werden
> sie **nicht offen veröffentlicht**; die Gutachterin erhält sie **unter
> Vertraulichkeitsvereinbarung (NDA)** separat. Gebäude, Betreiber, Standort
> und Zählernummern sind in dieser Fassung durchgängig anonymisiert; der
> Originaldatensatz heißt in der Dokumentation `gebaeude_a`.
>
> Das ist die von FitForFDM verlangte Trennung von Code und Daten: Der Code
> steht unter der [MIT-Lizenz](LICENSE), die Daten unterliegen einer
> **Zugriffsbeschränkung** statt einer offenen Datenlizenz.

### Der mitgelieferte Demo-Datensatz

Damit das Repository trotzdem vollständig lauffähig bleibt, erzeugt
[`src/tools/make_synthetic_data.py`](src/tools/make_synthetic_data.py) einen
**synthetischen Ersatzdatensatz** `demo_synthetic` — ein Befehl, rund 15
Sekunden, etwa 140 MB:

```bash
python src/tools/make_synthetic_data.py
```

Nachgebildet ist allein die **Form** der Originaldateien, denn genau daran
hängen die Lade- und Qualitätssicherungsschritte: Semikolon-Trenner, deutsches
Dezimalkomma, CRLF-Zeilenenden, zählerspezifische Nachkommastellen, die
Blockstruktur der Exporte samt Zählerneustart und Nullzeile, die realen
Dateigrenzen mit der mehrwöchigen Lücke im Spätsommer 2020 und die fehlende
Stunde bei jeder Sommerzeitumstellung.

**Alle Messwerte sind erfunden** — erzeugt aus einem Lastmodell mit festem
Zufallsstartwert (Jahres- und Tagesgang der Außentemperatur, daraus Heizbedarf;
drei Zähler mit bewusst verschiedenen Strukturprofilen). Übernommen wurden
weder Messwerte noch Zählernummern noch der Standort. Der Lauf ist
deterministisch: gleicher Startwert → byte-identische Dateien.

> **Die Demo reproduziert nicht die Ergebnisse der Arbeit.** Sie zeigt, dass
> die Pipeline durchläuft und plausible Strukturen findet — die Zahlen in
> [`docs/ergebnisbericht.md`](docs/ergebnisbericht.md) stammen aus den realen
> Daten und lassen sich ohne diese nicht nachrechnen.

### Datenherkunft und -dokumentation der Originaldaten (nach FitForFDM)

| | |
|---|---|
| **Wer hat erhoben** | Betriebsdaten der Wärmeversorgung, bereitgestellt vom **Wärmeversorger** (anonymisiert) als Export der Gebäudeleittechnik. Nicht in einem Experiment erzeugt, sondern im laufenden Betrieb protokolliert |
| **Was** | Drei Wärmemengenzähler eines **Gewerbegebäudes in Berlin** (Standort anonymisiert), je Zähler ein kumulativer Zählerstand in **MWh** und eine Momentanleistung in **kW**; dazu stündliche Wetterdaten (Außentemperatur u. a.) der nächstgelegenen Station von **Open-Meteo** (Zippenfenig 2023, CC BY 4.0) |
| **Zeitraum** | Rohexporte 19.11.2019 – 01/2024 in fünf Zeitabschnitten. Die Pipeline schneidet in Stage 1 auf ein **exakt vierjähriges Fenster** zu: `[2019-11-19 00:00, 2023-11-19 00:00)` → 2.117.880 Minutenzeilen, nach der Stundenaggregation **35.064 Stunden** |
| **Auflösung** | Rohdaten minütlich, Analyse stündlich (Begründung in Abschnitt 3.1.2 der Arbeit) |
| **Formate & Größe** | 5 × CSV Zählerprotokoll (semikolon-getrennt, deutsches Dezimalkomma, Spaltenköpfe = Zählernummern) à 11–47 MB und 1 × CSV Open-Meteo (komma-getrennt, ~1 MB) — zusammen **rund 137 MB** |
| **Qualitätssicherung** | Zweistufiger Dual-Channel-Fehlerfilter in `src/stage2_preprocess.py`: Stufe 1 prüft MWh- und kW-Kanal getrennt auf physikalisch unmögliche Werte, Stufe 2 auf Konsistenz zwischen beiden. **Jede verworfene Stunde ist protokolliert** in `reports/stage2_flag_log.csv` und `stage2_gap_log.csv` — der Datenqualitätsbefund ist ein eigenes Ergebnis der Arbeit (Abschnitt 4.1), keine stille Vorverarbeitung |
| **Datenschutz** | **Keine personenbezogenen Daten.** Gemessen wird Gebäudeverbrauch auf Ebene dreier Versorgungskreise, keine Wohn- oder Einzelplatzebene, keine Rückschlüsse auf identifizierbare Personen. Die Beschränkung folgt aus dem Betriebs- und Geschäftsinteresse des Versorgers, nicht aus der DSGVO |
| **Ablageort** | **Code:** GitHub, MIT-lizenziert. **Daten:** nicht veröffentlicht, Weitergabe an die Gutachterin unter NDA |
| **Werkzeuge inkl. Versionen** | Python **3.12.13**; alle Pipeline-Abhängigkeiten exakt gepinnt in **[requirements-lock.txt](requirements-lock.txt)** (u. a. pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.8.0, statsmodels 0.14.6, ruptures 1.1.10, matplotlib 3.10.9). **PyTorch ist bewusst nicht gepinnt**, weil es maschinenspezifisch installiert wird (CUDA/CPU/arm64) — siehe [Setup](#setup) |

### Reproduzierbarkeit

Stages 1–6 und die Injektion (Stage 9) sind **bit-identisch** wiederholbar,
ebenso die klassischen Detektoren in Stage 7/8/10. Nicht bit-identisch ist
allein der LSTM-Autoencoder auf der GPU, weil die CUDA-Kernel
nicht-deterministisch sind. Die Zwischenstände jeder Stage bleiben als
Parquet-Datei liegen, sodass sich jeder Schritt einzeln nachvollziehen lässt.

## Architektur & Datenfluss

Die Verarbeitung ist eine **Kette von 10 Stages** (`src/stageN_*.py`). Jede
Stage liest die Parquet-Ausgabe der vorigen aus `outputs/<dataset>/parquet/`
und schreibt ihre eigene — dadurch ist jeder Schritt einzeln nachvollziehbar
und wiederholbar:

```
CSV (data/)
  └─ stage1  → stage1_raw_merged.parquet        (Minuten-Merge WMZ + Wetter)
  └─ stage2  → stage2_hourly.parquet            (Fehlerfilter, Bereinigung, stündlich)  + reports/stage2_*_log.csv
  └─ stage3  → stage3_stl.parquet               (MSTL: trend/seasonal/residual)
  └─ stage4  → stage4_features_{raw,residual}.parquet   (Feature Engineering)
  └─ stage5  → split_assignment.parquet         (zeitlicher Train/Val/Test-Split)
  └─ stage6  → stage6_normalized_{raw,residual,trend}.parquet   (StandardScaler je Variante)
  └─ stage7  → models/…                         (Training mit Default-HPs, Smoke)
  └─ stage8  → hpo/best_hparams.json, hpo/hpo_log.csv           (Hyperparameter-Optimierung)
  └─ stage9  → stage9_injected_{raw,residual,trend}.parquet, stage9_ground_truth.parquet
  └─ stage10 → reports/stage10_metrics.csv, reports/stage10_qualitative_*.csv  (Evaluation)
```

**Geteilte Kernmodule** (`src/`):

| Datei | Rolle |
|---|---|
| `models/base.py` | abstrakte `AnomalyDetector`-Basisklasse: einheitliches `fit(X)` / `score(X)`-Interface (höherer Score = anomaler); NaN-Konvention |
| `models/registry.py` | erzeugt die Trainings-Jobs `(Variante × WMZ × Modell)`; `model_features()` wählt die Features je Variante und schließt andere Zähler aus (Anti-Cross-Leakage) |
| `anomaly_injection.py` | Injektions-Bibliothek: Event-Planung + Anomalie-Typen (Spike/Drop/Plateau/Leckage/Drift/Strukturbruch) |
| `injection_apply.py` | wendet die Injektion an und re-normalisiert (von Stage 8 **und** 9 geteilt → identische Injektion in Val und Test) |
| `evaluation.py` | Metriken: `point_adjust`, `roc_auc`, `pr_auc`, `threshold_from_quantile`; Schwellwert = Quantil der sauberen Val-Scores |
| `result_io.py` | nicht-destruktives Schreiben/Mergen der Stage-8/10-Outputs |

Jeder Detektor erbt von `base.AnomalyDetector`; der LSTM-AE bildet gleitende
Fenster und behält nur NaN-freie Fenster. **Determinismus:** Stages 1–6 und
die klassischen Detektoren sind bit-identisch reproduzierbar; der
LSTM-Autoencoder auf GPU ist es nicht (nicht-deterministische CUDA-Kernel).

**Empfohlener Einstieg in den Code:**

1. Diese README; das Pipeline-Diagramm lässt sich mit
   `python src/tools/pipeline_diagram.py` erzeugen (visuelle Karte).
2. Einmal klein laufen lassen (Sekunden): `stage1`→`stage6`, dann
   `python src/stage7_train.py --max-train-rows 2000 --wmz wmz_1 --variants raw`.
3. `src/stageN_*.py` in Reihenfolge 1→10 lesen — jede Datei hat oben einen
   Docstring mit Ein-/Ausgabe.
4. Kernmodule (Tabelle oben), dann die einzelnen Detektoren in `src/models/`.
5. `src/tools/` + `src/exploration/` zuletzt — eigenständige Analyse-/
   Plot-Skripte, nicht Teil der Kern-Pipeline.

## Setup

Python 3.12. Virtuelle Umgebung anlegen und Abhängigkeiten installieren:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
```

**PyTorch** (nur für den LSTM-Autoencoder) wird maschinenspezifisch separat
installiert — die `requirements-lock.txt` enthält es bewusst nicht:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA-GPU (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only
```

> **Hinweis für Apple Silicon (M-Serie):** Ein natives **arm64**-venv verwenden
> (nicht Rosetta/x86 — das zieht ein defektes torch). Der LSTM-Autoencoder läuft
> dort **nicht** auf der GPU (MPS): PyTorchs `nn.LSTM`-Kernel ist auf MPS defekt
> (Speicher-Explosion) → auf dem Mac `--device cpu` verwenden.

In den folgenden Befehlen steht `python` für den Interpreter des aktivierten
venv (alternativ direkt `.venv/bin/python …` ohne Aktivierung).

## Pipeline selbst durchführen

Die zehn Stages bauen aufeinander auf und werden der Reihe nach ausgeführt.
Jede Stage liest die Parquet-Ausgabe der vorigen und schreibt ihre eigene nach
`outputs/<dataset>/`. **Ohne den LSTM-Autoencoder läuft die gesamte Kette
deterministisch und vollständig auf der CPU**, auf jedem Betriebssystem: Die
Datenstufen 1–6 und die Injektion (Stage 9) sind bit-identisch reproduzierbar,
die klassischen Detektoren in Stage 7/8/10 ebenso. Nur der LSTM-Autoencoder
profitiert von einer GPU — und ist dort **nicht** bit-identisch wiederholbar,
weil die CUDA-Kernel nicht deterministisch sind (siehe Schritt B).

### Schritt 0 — Demo-Datensatz erzeugen

Die Originaldaten liegen dem Repository nicht bei (siehe [Daten](#daten)). Vor
dem ersten Lauf deshalb einmalig den synthetischen Ersatzdatensatz erzeugen:

```bash
python src/tools/make_synthetic_data.py     # ~15 s, schreibt data/demo_synthetic/
```

Danach ist `demo_synthetic` der Default-Datensatz aller Stages; ein eigener
Datensatz kommt über `--dataset <name>` dazu (siehe
[Eigenen Datensatz hinzufügen](#eigenen-datensatz-hinzufügen)).

### Schritt A — Pipeline ohne LSTM (vollständig auf CPU, ~Minuten)

```bash
python src/stage1_load.py         # CSVs laden + mergen, auf 4-Jahres-Fenster clippen
python src/stage2_preprocess.py   # Dual-Channel-Fehlerfilter -> stündliche Reihe
python src/stage3_stl.py          # MSTL-Zerlegung (Trend + Saison + Residuum)
python src/stage4_features.py     # Feature Engineering (Variante A=raw, B=residual)
python src/stage5_split.py        # zeitlicher Train/Val/Test-Split (kein Shuffle)
python src/stage6_normalize.py    # StandardScaler je Variante (fit nur auf Train)
python src/stage9_inject.py       # synthetische Anomalien ins Test-Set (Seed 99)

# klassische Detektoren (Z-Score, LOF, IForest, PELT, Constancy) — CPU, Sekunden:
python src/stage7_train.py        --models zscore lof iforest pelt constancy
python src/stage8_hpo.py          --models zscore lof iforest pelt constancy
python src/stage10_evaluate.py    --models zscore lof iforest pelt constancy
```

Reihenfolge-Hinweis: Stage 9 (Injektion) ist von den Modellen unabhängig und
kann jederzeit vor Stage 8/10 laufen. Stage 8 (HPO) braucht die injizierte
**Validierung**, Stage 10 das injizierte **Test-Set**.

### Schritt B — LSTM-Autoencoder (GPU empfohlen)

Der LSTM-AE ist das einzige Modell, das von einer GPU profitiert. Mit einer
NVIDIA-GPU (`--device cuda`) dauert der volle Lauf etwa ein bis zwei Stunden
(die Ergebnisse der Arbeit wurden auf einer RTX 3080 Ti trainiert); ohne GPU
läuft er mit `--device cpu` überall, nur deutlich langsamer. Stage 8/10
schreiben **nicht-destruktiv** — der LSTM-Lauf ergänzt also nur die LSTM-Zeilen
und lässt die klassischen Ergebnisse aus Schritt A unangetastet:

```bash
# mit NVIDIA-GPU; ohne GPU stattdessen --device cpu:
python src/stage7_train.py        --models lstm_ae                 # optionaler Smoke-Test (Device automatisch)
python src/stage8_hpo.py          --models lstm_ae --device cuda   # HPO (das eigentliche Training)
python src/stage10_evaluate.py    --models lstm_ae --device cuda   # Re-Fit mit besten HPs + Evaluation
```

(`stage7_train.py` kennt kein `--device` — es trainiert nur mit Default-HPs als
Smoke-Test und wählt das Backend automatisch; die belastbaren LSTM-Ergebnisse
liefern Stage 8 + 10.)

**Optional — Parallel-Lauf auf reiner CPU** (Bash-Skript; macOS/Linux, unter
Windows z. B. via WSL): Ein einzelnes LSTM-Training lastet faktisch nur einen
Kern aus. `src/tools/run_parallel_lstm.sh` startet deshalb alle neun
(Variante × WMZ)-Jobs gleichzeitig, jeden in einem isolierten Temp-Verzeichnis
`outputs/_par_<variante>_<wmz>/` und mit `OMP_NUM_THREADS=1` auf einen Kern
gepinnt (auf ~12 Kernen ca. 6–8× schneller als sequentiell).
Danach führt `python src/tools/merge_parallel.py` die Teilergebnisse einmalig
und nicht-destruktiv ins echte Dataset zusammen — die klassischen Ergebnisse
aus Schritt A bleiben unberührt. Ergebnisgleich zum sequentiellen Weg, nur
schneller; mit GPU ist der direkte `--device cuda`-Lauf oben vorzuziehen.

```bash
bash src/tools/run_parallel_lstm.sh     # 9 Jobs parallel (CPU)
python src/tools/merge_parallel.py     # Teilergebnisse zusammenführen
```

### Ergebnisse

Alles landet unter `outputs/<dataset>/` (gitignored, vollständig reproduzierbar).
Zentrale Datei: `reports/stage10_metrics.csv` (quantitativ, nach Anomalietyp und
-intensität stratifiziert, mit point-adjusted P/R/F1 + schwellenfrei ROC-AUC und
PR-AUC). Dazu qualitative Top-50-Score-Listen je Detektor zur manuellen Sichtung.

### Was im `outputs/`-Ordner liegt

Der Ordner wird **nicht** versioniert (`.gitignore`: `outputs/`) — Code plus die
Roh-CSVs unter `data/` genügen, um ihn vollständig neu zu erzeugen. Lokal sind
es rund 250 MB, im Repo liegt davon nichts.

```
outputs/<dataset>/
  parquet/   Zwischenstände zwischen den Stages (Transportformat)
  csv/       Klartext-Kopien zum Reinschauen ohne Python
  models/    trainierte Detektoren, <variante>/<wmz>/<modell>.pkl
  scalers/   Mittelwert und Streuung je Variante aus Stage 6
  hpo/       best_hparams.json + hpo_log.csv
  reports/   Metriken, Protokolle, Diagnose-Tabellen
  figures/   alle Abbildungen
```

**Die Ergebnisdateien** (~10 MB; alles andere ist reproduzierbarer
Zwischenstand):

| Datei | wofür |
|---|---|
| `reports/stage10_metrics.csv` | zentrale Ergebnistabelle: P/R/F1/ROC-AUC je Variante × WMZ × Modell × Stratum |
| `reports/stage10_qualitative_*.csv` | Top-Anomalien je Job mit Score (qualitative Sichtung) |
| `reports/stage10_changepoints_regulatory.csv` | PELT-Changepoints um den EnSikuMaV-Stichtag |
| `reports/data_sweep.csv` | Lernkurven-Punkte (F1 über Trainingsmenge) |
| `reports/stage2_flag_log.csv`, `stage2_gap_log.csv` | welche Stunde warum als fehlerhaft markiert wurde |
| `reports/stl_strength.csv`, `mutual_information_temp.csv` | Kennzahlen der Datenexploration |
| `hpo/best_hparams.json` | die in Stage 8 gewählten Hyperparameter je Job |

**Zu den Dateiformaten:**

- **`.parquet`** — spaltenorientiertes Binärformat, das Transportformat zwischen
  den Stages. Gegenüber CSV entscheidend ist nicht die Größe, sondern die
  **Typ-Treue**: Zeitstempel bleiben Zeitstempel, `NaN` bleibt `NaN`. Stage 3
  verlässt sich darauf, dass die Lückenmarkierungen aus Stage 2 exakt erhalten
  sind — bei einem Umweg über Text wären sie Verhandlungssache. Lesen mit
  `pd.read_parquet(...)`, nicht im Editor.
- **`.pkl`** — der trainierte Detektor mit seinem gelernten Zustand
  (`models/<variante>/<wmz>/<modell>.pkl`). Die klassischen Modelle nutzen
  Pythons `pickle`, der LSTM-Autoencoder intern `torch.save`; beide werden über
  dieselbe Schnittstelle geladen (`Detektor.load(pfad)`). Die Dateigrößen
  verraten die Modellnatur: `lof.pkl` ~7 MB (speichert die Trainingspunkte),
  `iforest.pkl` ~3 MB (die Baum-Ensembles), `lstm_ae.pkl` ~130 KB (nur
  Netz-Gewichte), `zscore.pkl` ~3 KB (ein paar Zahlen).
  *Sicherheitshinweis:* Das Laden einer Pickle-Datei kann beliebigen Code
  ausführen — nur selbst erzeugte `.pkl` laden.
- **`.csv`** — überall dort, wo ein Mensch oder Excel die Datei lesen soll,
  nicht für den Transport zwischen Stages.
- **`.json`** — nur `hpo/best_hparams.json`, verschachtelt nach
  `[variante][wmz][modell]`; Stage 10 liest daraus die finale Parametrisierung.

Dateien mit den Suffixen `_before_*`, `_classic` oder `_smoke` in `hpo/` und
`reports/` sind alte manuelle Snapshots von vor der nicht-destruktiven
Schreiblogik in Stage 8/10. Sie werden nicht mehr gebraucht und können weg.

### Weitere Auswertungen (Kennzahlen der Arbeit)

Diese Skripte erzeugen die übrigen in der Arbeit berichteten Zahlen; sie setzen
die durchlaufene Pipeline voraus (Schritt A, teils Schritt B):

```bash
python src/tools/lag_ablation.py                # Lag-Feature-Ablation (Abschnitt 3.1.5; nur Konsole)
python src/tools/seed_variance.py               # Inter-Seed-Streuung der LSTM-AE-HP-Wahl (Abschnitt 4.3; wertet hpo_log.csv aus, Stage 8 mit --hpo-seeds 3 nötig)
python src/tools/lstm_ae_score_ablation.py      # Score-Ablation des LSTM-AE, --device cuda empfohlen (Abschnitt 4.3)
python src/tools/qualitative_klassifikation.py  # Protokoll-Klassifikation der Top-50-Alarme (Abschnitt 4.6)
python src/tools/qualitative_protocol.py        # systematisches Fall-Protokoll zur qualitativen Evaluation (Abschnitt 3.5.2 / 4.6)
python src/tools/schwellen_sensitivitaet.py     # wie stark die Urteile an der Alarmschwelle haengen (Abschnitt 5.4, L9; setzt das Fall-Protokoll voraus)
python src/tools/pruefe_alarm_oekonomie.py      # stellt die Alarm-Ökonomie-Zahlen aus Abschnitt 4.5 neben die Werte in stage10_metrics.csv
```

`qualitative_protocol.py` ist die Umsetzung des qualitativen Leitfadens: Es
erzeugt die Recall-Matrix (Anomalietyp × Detektor), eine Event-Matrix je Zähler
mit Trefferurteil und Spitzen-Score relativ zur Schwelle, dazu die stärksten
Alarm-Segmente **außerhalb** der Ground-Truth-Events als
False-Positive-Kandidaten. Die klassischen Detektoren werden dafür
Stage-10-identisch neu gefittet (schnell); `--include-lstm` bezieht den
LSTM-Autoencoder ein (auf der CPU langsam). Zielpfad über `--out` frei wählbar.
Die fachliche Deutung der Fälle bleibt bewusst manuell — sie ist der
wissenschaftliche Beitrag, nicht das Skript.

`schwellen_sensitivitaet.py` wertet dieses Protokoll weiter aus: Jede
Trefferzeile trägt das Verhältnis *Spitzen-Score im Ereignis ÷ Schwelle*,
und daraus ergibt sich ohne jeden Neulauf, wie viel Reserve ein Treffer über
der Alarmschwelle hat. Das ist die Zahlenbasis für Limitation L9
(Abschnitt 5.4): Weil der Split chronologisch ist, wird die Schwelle in der
Heizsaison geeicht und im Sommerhalbjahr angewendet — je knapper die Treffer
über der Schwelle liegen, desto stärker wiegt das.

**Daten-Sweep** — Grundlage der Tabelle „point-adjusted F1 je Trainingsmenge"
(Abschnitt 4.3) *und* der Lernkurven-Abbildung (Abschnitt 3.4). Jeder Lauf
trainiert denselben Detektor auf 2.000 / 5.000 / 10.000 / allen Train-Stunden
und bewertet auf dem unveränderten Test-Set; die Ergebnisse werden kumulativ in
`reports/data_sweep.csv` gemergt (16 Kombinationen × 4 Stufen = 64 Zeilen).
`learning_curves.py` liest ausschließlich diese CSV — **ohne den Sweep bleibt
die Abbildung leer.**

```bash
for w in wmz_1 wmz_2 wmz_3; do
  python src/tools/data_sweep.py --variant raw      --wmz $w --model iforest
  python src/tools/data_sweep.py --variant residual --wmz $w --model lof
  python src/tools/data_sweep.py --variant trend    --wmz $w --model pelt
  python src/tools/data_sweep.py --variant raw      --wmz $w --model lstm_ae
  python src/tools/data_sweep.py --variant trend    --wmz $w --model lstm_ae
done
python src/tools/data_sweep.py --variant residual --wmz wmz_1 --model lstm_ae
```

> Die Hyperparameter kommen je Lauf aus `hpo/best_hparams.json`, damit nur die
> Trainingsmenge variiert. `data_sweep.py` kennt bewusst kein `--device`: Das
> Backend wählt der Detektor selbst (CUDA, sonst CPU). Die Klassik-Läufe
> (IForest/LOF/PELT) brauchen auf der CPU Minuten; die sechs LSTM-AE-Läufe
> gehören auf die GPU.

## Abbildungen der Arbeit reproduzieren

**Jede Abbildung der Arbeit wird von genau einem Skript erzeugt**, ausschließlich
aus den Rohdaten und den Pipeline-Outputs — es liegen bewusst keine fertigen
Bilddateien im Repo. Reproweg: **erst die Pipeline durchlaufen** (Abschnitt oben,
mindestens Schritt A; für `score_verteilungen` und die LSTM-Fallplots auch
Schritt B), **dann die Plot-Skripte ausführen.** Alle Abbildungen landen in
`outputs/demo_synthetic/figures/` (Ausnahme: das Pipeline-Diagramm → `docs/`).

```bash
# Explorative Analyse
for s in temperature_scatter hour_weekday_heatmap autocorrelation \
         monthly_boxplot load_duration_curve cross_meter_correlation \
         temperature_crosscorrelation mutual_information stl_strength \
         temperature_regimes; do
  python src/exploration/$s.py
done

# Stage-bezogene Plots
python src/tools/stage2_explore.py        # tagesprofil, monatsprofil, temperatur_korrelation, glitch_zeitverlauf
python src/tools/stage3_explore.py        # stl_zoom_*, stl_trend_alle, stl_tagesprofil, stl_wochenprofil, stl_residual_hist
python src/tools/stage3_macroevents.py    # monatlich_makro_events, heizsaison_vergleich
python src/tools/plot_injektionskarte.py  # injektionskarte
python src/tools/learning_curves.py       # learning_curves        (setzt data_sweep.csv voraus, s. o.)
python src/tools/score_distributions.py   # score_verteilungen     (--device cuda möglich)
python src/tools/plot_recall_heatmap.py   # recall_heatmap
python src/tools/plot_qualitative_case.py # qualitative Fallplots -> figures/qualitativ/

# Konzept-Schaubilder (keine Daten, laufen ohne Pipeline)
python src/tools/pipeline_diagram.py      # docs/pipeline.png
python src/tools/plot_ki_schachtelung.py  # ki_schachtelung
python src/tools/anomalietypen_schema.py  # anomalietypen_schema

# LSTM-AE-Diagnostik (setzt trainierte Modelle voraus, Schritt B)
python src/tools/lstm_ae_reconstruction.py           # lstm_ae_reconstruction_raw_wmz_2
python src/tools/lstm_ae_reconstruction.py --latent  # zusaetzlich lstm_ae_latent_raw_wmz_2
```

**Zuordnung Abbildung → Skript:**

| Abbildung(en) | erzeugendes Skript |
|---|---|
| `tagesprofil`, `monatsprofil`, `temperatur_korrelation`, `glitch_zeitverlauf` | `src/tools/stage2_explore.py` |
| `stl_zoom_wmz_{1,2,3}`, `stl_trend_alle`, `stl_tagesprofil`, `stl_wochenprofil`, `stl_residual_hist` | `src/tools/stage3_explore.py` |
| `monatlich_makro_events`, `heizsaison_vergleich` | `src/tools/stage3_macroevents.py` |
| `temperatur_scatter_lowess` | `src/exploration/temperature_scatter.py` |
| `heatmap_stunde_wochentag` | `src/exploration/hour_weekday_heatmap.py` |
| `autokorrelation_kw` | `src/exploration/autocorrelation.py` |
| `boxplot_monatlich` | `src/exploration/monthly_boxplot.py` |
| `jahresdauerlinie` | `src/exploration/load_duration_curve.py` |
| `cross_meter_correlation` | `src/exploration/cross_meter_correlation.py` |
| `kreuzkorrelation_temp` | `src/exploration/temperature_crosscorrelation.py` |
| `mutual_information_temp` | `src/exploration/mutual_information.py` |
| `temperatur_regime` | `src/exploration/temperature_regimes.py` |
| `stl_strength` | `src/exploration/stl_strength.py` |
| `injektionskarte` | `src/tools/plot_injektionskarte.py` |
| `learning_curves` | `src/tools/learning_curves.py` |
| `score_verteilungen` | `src/tools/score_distributions.py` |
| `recall_heatmap` | `src/tools/plot_recall_heatmap.py` |
| `lstm_ae_reconstruction_<variante>_<wmz>`, `lstm_ae_latent_<variante>_<wmz>` (`--latent`) | `src/tools/lstm_ae_reconstruction.py` |
| `pipeline` (→ `docs/`) | `src/tools/pipeline_diagram.py` |
| `ki_schachtelung` (→ `docs/`) | `src/tools/plot_ki_schachtelung.py` |
| `anomalietypen_schema` | `src/tools/anomalietypen_schema.py` |
| qualitative Fallplots (`figures/qualitativ/*`) | `src/tools/plot_qualitative_case.py` |

> `score_verteilungen`, die LSTM-AE-Diagnostik und die qualitativen
> LSTM-AE-Fallplots setzen trainierte Modelle voraus (Schritt B); alle übrigen
> Abbildungen laufen auf der CPU. Die beiden LSTM-AE-Diagnose-Abbildungen der
> Arbeit zeigen `raw`/`wmz_2` — das sind die Defaults, ein nackter Aufruf
> genügt.

Die **qualitativen Fallplots** entstehen je Kombination aus Variante, Zähler,
Detektor und injiziertem Ereignis; ein Plot zeigt das Signal in kW, das
Ground-Truth-Band des Ereignisses sowie den Detektor-Score mit seiner Schwelle.
`--list` gibt statt eines Plots alle Ereignisse dieses Jobs samt TP/FN-Urteil
aus — das ist der Weg zur Fallauswahl. Variante, Zähler und Modell sind dabei
immer anzugeben, auch beim Auflisten:

```bash
python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --list
python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --type spike --intensity 10
```

## Nützliche Optionen (alle Stages)

- `--dataset <name>` — anderer Datensatz (Default `demo_synthetic`).
- `--models <a b c>` (Stage 7/8/10) — Detektor-Teilmenge.
- `--device cpu|cuda|mps|auto` (LSTM, Stage 7/8/10) — Rechen-Backend.
- `--max-train-rows N` (Stage 7/8/10) — kleinerer Trainings-Tail für schnelle
  Probeläufe ohne GPU.
- `--fresh` (Stage 8/10) — vorhandene Ergebnisse **überschreiben** statt
  nicht-destruktiv zu mergen.

## Eigenen Datensatz hinzufügen

1. Unterordner `data/<name>/` anlegen.
2. Die Logging-CSVs (Zeitabschnitte des Gebäudes) nach dem Muster
   `logging_heat-energy_*.csv` ablegen, die Wetterdatei als `open-meteo-*.csv`.
   Stage 1 findet sie über diese Namensmuster; die Zählernummern in den
   Spaltenköpfen sind beliebig, ihre **Reihenfolge** legt `wmz_1`, `wmz_2`, …
   fest. Als Formatvorlage dient `data/demo_synthetic/` aus Schritt 0.
3. Pipeline mit `--dataset <name>` aufrufen. Outputs landen unter
   `outputs/<name>/`; bestehende Datensätze bleiben unberührt.

## Projektstruktur

```
src/
  stage{1..10}_*.py     Pipeline-Stages
  anomaly_injection.py  Injektions-Bibliothek (Stage 8/9)
  injection_apply.py    Injektion anwenden + re-normalisieren
  evaluation.py         Metriken (point-adjust, ROC-AUC)
  result_io.py          Nicht-destruktives Schreiben der Stage-8/10-Ergebnisse
  models/               Detektoren (base, zscore, lof, iforest, constancy, pelt, lstm_ae, registry)
  tools/                Stage-bezogene Hilfs- und Plot-Skripte
    make_synthetic_data.py  Generator des Demo-Datensatzes (Schritt 0)
  exploration/          Methoden-zentrierte Explorations-/Abbildungs-Skripte
docs/                   Methodik-, Ergebnis- und Auswertungsdokumentation
data/<dataset>/         Eingabedaten (WMZ-CSVs + Open-Meteo) — gitignored.
                        `demo_synthetic` erzeugt Schritt 0; die Originaldaten
                        sind nicht Teil der Veröffentlichung (s. "Daten")
outputs/<dataset>/      Generierte Artefakte (parquet/csv/reports/figures/
                        scalers/models/hpo) — gitignored, vollständig reproduzierbar
```

## Kernbefund

**Mustererkennung:** Das Strukturprofil aus der MSTL-Zerlegung ist ein
**Fingerabdruck der realen Versorgungsfunktion** — trend-dominierte
Raumheizung (Strength 0,82), stark tages-/wochenperiodische Küche (0,73) und
schwach strukturiertes Allgemein-Warmwasser (0,30); das deckt sich mit der
Betreiberauskunft und belegt die Zerlegung physikalisch. Praktisch nutzbar
ist es, weil das Profil **vorhersagt, welche Detektions-Schiene je Zähler
trägt**. Die Außentemperatur wirkt dabei **nichtlinear** (Knick an der
Heizgrenze mit Sättigung): Zwei Zähler mit praktisch gleicher
Pearson-Korrelation unterscheiden sich in der Mutual Information um
Faktor 1,9 — ein linearer Kennwert allein hätte sie gleich aussehen lassen.

**Anomaliedetektion:** Auf diesem einfachen, periodischen Regime erreichen
die **klassischen Detektoren** die beste Leistung (Isolation Forest
F1 = 0,91); der LSTM-Autoencoder bleibt confound-frei und kriteriums-robust
darunter. Ein KI-Mehrwert ist erst bei heterogeneren Daten zu erwarten.
Sämtliche Pipeline-Stages und Detektoren sind implementiert und ausgewertet;
das LSTM-Training auf der GPU (RTX 3080 Ti) ist abgeschlossen.

Die naheliegende Fortsetzung — eine Robustheitsprüfung auf einem zweiten
Gebäude — erfordert **keinen neuen Code**, sondern nur einen weiteren Datensatz
unter `data/` und einen Lauf mit `--dataset` (s. o.).

## Unterschiede zur eingereichten Fassung

Dies ist die **öffentliche Fassung** des Repositories. Sie unterscheidet sich
von der bei der Hochschule eingereichten in sechs Punkten:

| | |
|---|---|
| **Keine Originaldaten** | Statt der vertraulichen Messreihen liegt ein Generator für einen synthetischen Ersatzdatensatz bei (siehe [Daten](#daten)) |
| **Anonymisiert** | Gebäude, Betreiber, Standort und Zählernummern sind ersetzt; die Dokumentation nennt den Originaldatensatz `gebaeude_a` |
| **Eine funktionale Änderung** | `stage1_load.py` findet die Eingabedateien über die Namensmuster `logging_heat-energy_*.csv` und `open-meteo-*.csv` statt über eine fest verdrahtete Dateiliste, und leitet `wmz_1/2/3` aus der Spaltenreihenfolge ab statt aus konkreten Zählernummern. Für den Originaldatensatz ist das Ergebnis identisch |
| **Keine Ergebnisartefakte** | `outputs/` ist leer — die Zahlen und Abbildungen der Arbeit stammen aus den Originaldaten und lassen sich ohne diese nicht erzeugen |
| **Bibliografie gekürzt** | Aus [`docs/literatur.bib`](docs/literatur.bib) sind vier Webquellen entfernt, die den Standort identifizieren, sowie die internen Arbeitskommentare |
| **Interne Dokumente fehlen** | Arbeitsprotokoll, Kolloquiumsvorbereitung, Quellenprüfung und die Arbeit selbst sind nicht enthalten; Querverweise darauf in `docs/` laufen deshalb ins Leere |

## Lizenz und Zitation

Der **Code** steht unter der [MIT-Lizenz](LICENSE) — frei nutzbar, auch
kommerziell, unter Nennung des Copyright-Hinweises.

Die **Daten** sind davon ausgenommen: Sie stehen unter keiner offenen Lizenz,
sondern unterliegen einer Zugriffsbeschränkung (Abschnitt [Daten](#daten)).
Wer den Code auf eigene Daten anwenden will, braucht deshalb keine Erlaubnis;
wer die Ergebnisse dieser Arbeit exakt nachrechnen will, braucht den Datensatz
und damit die Zustimmung des Datengebers.

Zitationsvorschlag für die Arbeit:

> Riedel, L. M. (2026): *Potenzialanalyse von KI für Mustererkennung und
> Anomaliedetektion von Energieverbrauchsdaten.* Masterarbeit,
> Hochschule für Technik und Wirtschaft Berlin, Fachbereich 4,
> M. Sc. Wirtschaftsinformatik.
