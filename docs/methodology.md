# Methodik — Pipeline und Datenaufbereitung

> **Hinweis zur veröffentlichten Fassung.** Alle hier berichteten Zahlen
> stammen aus den **realen Betriebsdaten**, die nicht Teil dieses Repositories
> sind (siehe [README, Abschnitt „Daten"](../README.md#daten)). Gebäude,
> Betreiber, Standort und Zählernummern sind anonymisiert; der Datensatz heißt
> hier durchgängig `gebaeude_a`. Der mitgelieferte Demo-Datensatz
> `demo_synthetic` enthält **synthetische Werte** und reproduziert diese Zahlen
> nicht — er hält allein die Pipeline lauffähig.
>
> Verweise auf `HANDOFF.md` und `MA.docx` betreffen das interne
> Arbeitsprotokoll bzw. die Arbeit selbst; beide sind nicht Teil dieses
> Repositories.

Dieses Dokument hält die Pipeline-Architektur sowie die konkret
umgesetzten Aufbereitungsschritte auf dem sauberen Gebaeude-A-Datensatz
fest. Jede Entscheidung ist mit Begründung, Parameter und
quantitativem Effekt dokumentiert, damit sie in der Masterarbeit
reproduzierbar zitiert werden kann.

**Sektion 1** beschreibt die Rohdaten. **Sektionen 2–4** dokumentieren
die **implementierten** Stages 1, 2 und 3 (Laden, Vorverarbeitung mit
Dual-Channel-Fehlerfilter, MSTL-Dekomposition). **Sektion 5** fasst
empirische Befunde aus den implementierten Stages zusammen.
**Sektion 6** stellt die übergeordnete **Pipeline-Architektur** vor —
Anomalie-Kategorien, drei Detektions-Varianten A/B/C und
Trainings-Filterung. **Sektion 7** dokumentiert die **implementierten
Stages 4–10** (Feature Engineering bis Evaluation) sowie **Stage 11**
(Robustheits-Evaluation, kein neuer Code nötig). **Sektion 8**
beschreibt die Reproduzierbarkeit und wie weitere Datensätze
hinzugefügt werden können.

**Implementierungsstand (Stand der Code-Basis):** Die Stages 1–10 sind
vollständig implementiert und durchgelaufen — inklusive aller fünf
Detektoren (Z-Score, LOF, Isolation Forest, PELT, LSTM-Autoencoder),
Hyperparameter-Optimierung, synthetischer Anomalie-Injektion und
stratifizierter Evaluation. Stage 11 erfordert keinen neuen Code,
sondern nur einen erneuten Pipeline-Lauf auf einem zweiten Datensatz
(`--dataset`), der derzeit noch nicht vorliegt.

---

## 1. Rohdatenlage

Quelle: Gebäude A, 5 CSV-Dateien aus dem
Datenerfassungssystem mit drei Wärmemengenzählern (WMZ), Sampling
~1 Minute, sowie eine Open-Meteo-Wetterzeitreihe (stündlich,
Standort 52.41 N / 13.53 E, Höhe 34 m, lokale Europe/Berlin-Zeit).

| Datei | Zeitraum |
|---|---|
| logging_heat-energy_gebaeude-a_2019.11-2020.08.csv | 2019-11-19 07:14 — 2020-08-27 13:32 |
| logging_heat-energy_gebaeude-a_2020.09-2021.01.csv | 2020-09-10 08:36 — 2021-01-07 06:13 |
| logging_heat-energy_gebaeude-a_2021.01-2021.08.csv | 2021-01-07 06:15 — 2021-08-16 15:23 |
| logging_heat-energy_gebaeude-a_2021.08-2022.08.csv | 2021-08-16 15:29 — 2022-08-31 09:27 |
| logging_heat-energy_gebaeude-a_2022.08-2024.01.csv | 2022-08-31 09:31 — 2024-01-10 14:37 |
| open-meteo-station.csv | 2019-11-19 00:00 — 2023-11-19 23:00 |

Format-Eigenheiten der WMZ-CSVs:
- Trennzeichen `;`, Dezimaltrenner `,` (deutsches Locale)
- Datums-/Zeitformat `DD.MM.YYYY HH:MM:SS`, getrennt in zwei Spalten
- Spaltennamen enthalten die Zähler-IDs: `10000001 / MWh`,
  `10000001 / kW`, `10000002 / MWh`, `10000002 / kW`,
  `10000003 / MWh`, `10000003 / kW`
- Eine erste Hilfsspalte `No.` (Zeilennummer) und eine durch das
  abschließende `;` erzeugte leere Spalte am Zeilenende

Format-Eigenheiten der Wetter-CSV:
- 3 Zeilen Metadaten-Header (Koordinaten, Zeitzone) vor dem
  eigentlichen Datenheader
- Spaltennamen mit Einheit in Klammern: `temperature_2m (°C)`,
  `relative_humidity_2m (%)`, `precipitation (mm)`

---

## 2. Stage 1 — Laden und Zusammenführen

Implementiert in [src/stage1_load.py](../src/stage1_load.py).

| Schritt | Beschreibung | Wirkung |
|---|---|---|
| 1.1 | CSV-Parsing der 5 WMZ-Dateien mit `sep=";"`, `decimal=","`, `date_format="%d.%m.%Y %H:%M:%S"` | Dezimaltrenner konvertiert, Datetime geparst |
| 1.2 | `Date` + `Time` zu einer `timestamp`-Spalte kombiniert; Index gesetzt | Pandas-konformer DatetimeIndex |
| 1.3 | Spalten `No.` und durch trailing `;` erzeugte unbenannte Spalte entfernt | Hilfsspalten weg |
| 1.4 | Zählerspalten umbenannt: `10000001 → wmz_1`, `10000002 → wmz_2`, `10000003 → wmz_3` mit Suffix `_mwh` / `_kw` | Lesbare Namen, konsistente Konvention |
| 1.5 | Wetter-CSV geladen (`skiprows=3`); Spalten zu `temperature`, `humidity`, `precipitation` umbenannt | Einheitliche Spaltennamen |
| 1.6 | 5 WMZ-DataFrames konkateniert, chronologisch sortiert | 2 158 814 Zeilen |
| 1.7 | Doppelte Zeitstempel entfernt (`keep="first"`) | **240** Duplikate (Datei-Übergänge) → 2 158 574 |
| 1.8 | Outer-Join WMZ × Wetter auf `timestamp` | Gemeinsamer Index, fehlende Werte als NaN |
| 1.9 | Clipping auf `[2019-11-19 00:00, 2023-11-19 00:00)` (linksgeschlossen, **exklusives Ende**) | 4-Jahres-Fenster mit exakt 1 461 Tagen (inkl. Schaltjahr-Tag); **75 782** Zeilen verworfen |

**Output:** `outputs/<dataset>/parquet/stage1_raw_merged.parquet`
(2 117 880 × 9: 3 × MWh kumuliert, 3 × kW Momentan, `temperature`,
`humidity`, `precipitation`).

**In Stage 1 sichtbar geworden, aber nicht behoben:** Datenlücke
2020-08-27 13:33 → 2020-09-10 08:35 (~14 Tage). Diese erscheint in
Stage 2 als 331-h-NaN-Run.

---

## 3. Stage 2 — Vorverarbeitung (Dual-Channel-Fehlerfilter)

Implementiert in [src/stage2_preprocess.py](../src/stage2_preprocess.py).

Das primäre Signal nach Stage 2 ist `kW` (Stunden-Mittelwert), nicht
mehr `kWh-Delta`. Damit entfällt die Differenzbildung mit ihren
NaN-Propagations-Effekten und die Information aus beiden
Messkanälen des Wärmemengenzählers (MWh kumuliert und kW momentan)
wird sinnvoll genutzt: Fehler werden auf MWh erkannt (wo sie klare
Signaturen haben), die Bereinigung erfolgt auf kW (dem Modell-Input).

Die regelbasierten Detektoren stützen sich auf robuste, **median-basierte**
Statistik (gleitender Median, Cummax — kein literales MAD) — ein in der
Zeitreihen-Anomaliedetektion etabliertes Vorgehen, dessen median-/MAD-Prinzip
Vallis u. a. (2014) und Blázquez-García u. a. (2021) belegen.

**Warum Median und nicht MAD?** Leys u. a. (2013) empfehlen Median *und* MAD
statt Mittelwert/Std. Hier wird bewusst nur der **Median** (als robuste
Basislinie) übernommen, der Abweichungs-Schwellwert aber aus Domäne/Physik
gesetzt — aus vier Gründen:
1. **Robustheit kommt vom Median, nicht vom Streuungsmaß.** Leys' Kernargument
   ist das *Masking* (Mittelwert/Std werden von den Ausreißern selbst
   verzerrt). Der gleitende Median hat 50 % Breakdown-Punkt und ist gegen
   genau diese Glitches immun — die geforderte Robustheit ist damit erreicht.
2. **Die Schwellen sind physikalisch motiviert, nicht statistisch.** Die
   MWh-Reihe hat 1-kWh-Auflösung (0,001 MWh); eine 1,0-MWh-Sprungschwelle ist
   eine physikalisch bedeutsame Glitch-Größe, der kW-Spike läuft über
   *Faktor × lokaler Median + Sockel*. Das sind interpretierbare, auditierbare
   Domänen-Konstanten, kein Verteilungsmaß.
3. **MAD degeneriert in den dominanten Flachphasen.** wmz_3 ist im Sommer
   faktisch aus (~31 % nahe null), Wochenenden flach. Dort geht lokales
   MAD → 0, sodass „Median ± k·MAD" kollabiert → massives Over-Flagging
   legitimer Ruhephasen. Genau das verhindert der feste Sockel; MAD würde den
   gelösten Edge-Case wieder einführen.
4. **Teile der Logik sind kein Streuungsproblem.** Counter-Reset (Cummax auf
   kumuliertem MWh) und der Cross-channel-Guard sind regel-/physikbasiert —
   MAD ist dort nicht anwendbar.

### 3.1 Spaltenauswahl

- **Behalten:** `wmz_{1,2,3}_mwh`, `wmz_{1,2,3}_kw` (beide für die
  Fehler-Detektion), `temperature`, `humidity`
- **Entfernt:** `precipitation` — nicht im Feature-Plan

### 3.2 Stufe 1 — Fehler-Detektion (minutenweise, pro Signal)

Die Detektion läuft nur auf **WMZ-relevanten Zeilen** (gefiltert nach
„mindestens eine WMZ-Spalte nicht NaN"). Weather-only-Zeilen aus dem
Outer-Join in Stage 1 enthalten WMZ-NaN, was kein Daten-Ausfall ist.

**Sechs Fehler-Detektoren**, alle als Boolean-Flags pro Zeitstempel:

| Flag | Signal | Logik |
|---|---|---|
| `is_mwh_reset` | MWh | Wert fällt unter eigenes laufendes Maximum minus 1.0 MWh. Spike-up wird vorher maskiert, damit ein einzelner Aufwärts-Glitch das Maximum nicht „vergiftet" |
| `is_mwh_spike` | MWh | Wert über lokalem Median (Fenster 201 min) plus 1.0 MWh |
| `is_mwh_plateau` | MWh + kW (cross-channel) | MWh-Zuwachs über 60-min-Fenster ist viel kleiner als die kW-basierte Erwartung; zusätzlich `kW > 1` zur Vermeidung von Off-Phasen-False-Positives. **Erkennt:** MWh-Sensor stuck, während Heizung läuft |
| `is_kw_negative` | kW | `kW < 0` (physikalisch unmöglich) |
| `is_kw_spike` | kW | Über *zwei* Schwellen gleichzeitig: lokaler Median × 10 + Sockel, und globales 95-Perzentil × 3. Verhindert False Positives in Sommer-Off-Phasen, wenn der lokale Median nahe 0 liegt |
| `is_kw_plateau` | kW + MWh (cross-channel) | kW-Std ≈ 0 über 6-min-Fenster UND MWh ist im Fenster gestiegen. **Erkennt:** kW-Sensor stuck während Verbrauch tatsächlich stattfindet |
| `is_gap` | MWh und kW | Beide Signale gleichzeitig NaN — echter Daten-Ausfall |

**Wichtige Designentscheidungen:**

- Plateau-Detektoren sind **cross-channel-aware**: ein Sensor-Plateau
  wird nur geflaggt, wenn der jeweils andere Kanal Aktivität zeigt.
  Damit fallen echte Off-Phasen (Sommer-Heizung-aus für wmz_3)
  korrekterweise nicht in die Plateau-Kategorie.
- MWh-Plateau-Detektor verwendet ein **60-min-Fenster**, weil die
  kumulierte MWh nur mit 3 Nachkommastellen (0,001 MWh = 1 kWh)
  gemessen wird. Bei geringen Lastraten (z. B. wmz_1 mit ~10 kW)
  passiert ~0,000167 MWh/min — kürzere Fenster sind durch
  Mess-Präzision dominiert.
- `is_kw_negative` ist eine **Erweiterung gegenüber der ursprünglichen
  Spec** — wurde notwendig, weil wmz_3 vereinzelt negative kW-Werte
  liefert (klares Sensor-Versagen, in stündlichen Mitteln vor der
  Behebung als negative Verbräuche sichtbar).

**Effekt auf den Datensatz (Minuten-Level-Flags):**

| Zähler | reset | mwh_spike | mwh_plateau | kw_neg | kw_spike | kw_plateau | gap |
|---|---|---|---|---|---|---|---|
| wmz_1 | 6 303 | 0 | 43 | 0 | 2 | 517 | 0 |
| wmz_2 | 1 781 | 0 | 5 | 0 | 102 | 21 | 0 |
| wmz_3 | 157 287 | 105 | 416 031 | 2 648 | 0 | 1 611 | 0 |

Die hohe `mwh_plateau`-Zahl bei wmz_3 reflektiert die ~120 Tage
Sensor-Offline-Phasen korrekt — der MWh-Sensor ist über lange
Zeiträume stuck, während der kW-Kanal Aktivität zeigt. Genau dieses
Muster zu identifizieren ist Ziel des Detektors.

### 3.3 Stufe 2 — Bereinigung (auf kW)

- Jeder Zeitstempel mit irgendeinem aktiven Flag → NaN in `kW`
- Kurz-Lücken im kW-Signal (≤ 3 h = 180 Minuten) linear interpoliert
- Stündliches Resampling: `mean(kW)` pro Stunde
- Flags ebenfalls aggregiert: pro Stunde Anzahl Flag-Minuten je Typ
  plus eine `was_flagged`-Boolean-Summary
- Wetter wird unverändert auf Stunden gemittelt

**Warum stündlich (Auflösungs-Entscheidung, H5).** Die gesamte Pipeline
arbeitet bewusst auf Stunden-Auflösung. Das ist methodisch begründet und
nicht bloß pragmatisch: (1) Die ML-Zielanomalien (Kategorie B — Leckage,
Drift, Plateau, Strukturbruch) sind **thermisch träge** und wirken über
Stunden bis Tage (Frederiksen und Werner 2013; Lindberg u. a. 2019); (2) die
nachweislich sub-stündlichen Ereignisse sind **Sensorfehler (Kategorie A)**,
die genau hier — auf Minuten-Ebene, regelbasiert — erfasst und herausgefiltert
werden, also bewusst nicht das ML-Ziel sind; (3) Sub-Minuten-Smart-Meter-Daten
werden in der Domäne ohnehin zur Rausch-/Rechenreduktion auf Stunden
aggregiert (Amirkhanova u. a. 2026; Himeur u. a. 2021), und WMZ integrieren
Energie ohnehin (EN 1434 2015). Der Auflösungsvergleich (H5) wird deshalb
**analytisch** beantwortet statt über einen separaten Minuten-Lauf; ein
solcher lohnte nur mit echten sub-stündlichen Fehlersignaturen
(Ventil-Pendeln/Kurztaktung, Katipamula und Brambley 2005) — siehe
Gliederung § 4.4.3 und Ausblick.

### 3.4 Stage-2-Output-Schema

`outputs/<dataset>/parquet/stage2_hourly.parquet` — **35 064 × 32**:

| Spaltentyp | Anzahl | Beispiele |
|---|---|---|
| Wert-Signale | 5 | `wmz_{1,2,3}_kw_mean`, `temperature`, `humidity` |
| Interpolations-Flags | 3 | `wmz_{1,2,3}_interpolated` (bool) |
| Fehler-Flag-Summary | 3 | `wmz_{1,2,3}_was_flagged` (bool) |
| Fehler-Flag-Counts | 21 | `wmz_{N}_is_{type}_count` (int32, 7 Typen × 3 Zähler) |

**kW-Mean Verteilung** (NaN ausgeschlossen):

| | min | median | p99 | max |
|---|---|---|---|---|
| wmz_1_kw_mean | 0 | 9,9 | 13,3 | 27,8 |
| wmz_2_kw_mean | 0 | 4,9 | 53,8 | 78,9 |
| wmz_3_kw_mean | 0 | 54,6 | 484 | 853 |

**Flag-Summary** (Stunden mit irgendeinem aktiven Flag):

| Zähler | was_flagged | Anteil | interpoliert |
|---|---|---|---|
| wmz_1 | 5 564 | 15,9 % | 5 564 |
| wmz_2 | 2 144 | 6,1 % | 2 144 |
| wmz_3 | 18 559 | 52,9 % | 13 968 |

### 3.5 Logs und Reports

Persistiert nach `outputs/<dataset>/reports/`:

- `stage2_flag_log.csv` — pro Stunde alle Flag-Counts (kompakt, nur
  Stunden mit ≥ 1 Flag)
- `stage2_gap_log.csv` — Stunden mit Daten-Ausfällen

### 3.6 Anomalie-Kategorien und Flag-Verwendung

Die Stage-2-Flags sind **Ground-Truth-Quelle für Kategorie A**
(Datenfehler, siehe [Sektion 7](#6-pipeline-architektur-übergreifend)).
Sie werden in Stage 6 zur Trainings-Daten-Filterung und in Stage 10 zur
stratifizierten Evaluation verwendet, nicht aber als Modell-Input
während des Trainings.

### 3.7 Historisches — frühere v1-Implementation

Vor der Umstellung auf kW als Primärsignal arbeitete Stage 2 auf
kWh-Deltas (`diff(MWh) × 1000`) mit einem zweistufigen Glitch-Filter
(Rolling-Median + Cummax). Diese Version ist im Git-Verlauf vor Commit
`4efd35c` einsehbar; sie wurde durch die hier beschriebene
Dual-Channel-Architektur ersetzt, weil:

1. kW direkter zugänglich ist (kein Differenzbildungs-Artefakt)
2. Der zweite Messkanal die Diagnose von Sensor-Fehlern wesentlich
   verbessert (Konsistenz zwischen MWh und kW)
3. Die Detektor-Taxonomie expliziter und feinkörniger geworden ist
   (sechs Flag-Typen statt einem generischen Glitch-Flag)

## 4. Stage 3 — MSTL-Dekomposition

Implementiert in [src/stage3_stl.py](../src/stage3_stl.py).

### 4.1 Motivation und Verwendungsabsicht

Eine Multi-Seasonal-Trend-Decomposition (MSTL) zerlegt jede WMZ-Zeitreihe
in vier additive Komponenten:

```
kwh(t) = trend(t) + seasonal_24h(t) + seasonal_168h(t) + residual(t)
```

- **Trend** und **Saisonalitätskomponenten** dienen der
  **deskriptiven Mustererkennung**: typisches Tagesprofil,
  Werktag/Wochenend-Modulation, Jahresgang (im Trend absorbiert)
  und langsame Verschiebungen über die Beobachtungsjahre.
- **Residuum** ist der **primäre Eingang für die Anomaliedetektion**
  in den Stages 6–9. Modelle wie Z-Score, LOF, Isolation Forest und
  LSTM-Autoencoder werden auf dieser deseasonalisierten Größe
  trainiert, sodass sie nicht die ohnehin offensichtliche Periodizität
  lernen müssen, sondern direkt Abweichungen vom erwarteten Muster
  modellieren. Dieses Vorgehen — saisonale Dekomposition, anschließend
  Detektion auf dem Residuum — ist in der Zeitreihen-Anomaliedetektion
  etabliert (Hochenbaum u. a. 2017; Blázquez-García u. a. 2021).

**Implikation für Anomalietypen:** Spike, Drop und Plateau-Anomalien
manifestieren sich im Residuum. Drift und Strukturbruch werden
hingegen vom Trend absorbiert und sind durch reine Residuum-Modelle
*nicht* erkennbar — sie müssen durch separate Analyse der
Trend-Komponente identifiziert werden.

### 4.2 Verfahren und Parameter

- **MSTL** aus `statsmodels.tsa.seasonal` mit Perioden `[24, 168]`
  (Tag und Woche). Das Verfahren erweitert das klassische STL
  (Cleveland u. a. 1990) auf mehrere Saisonalitäten (Bandara u. a. 2025).
- **Robust = True** (`stl_kwargs={"robust": True}`): iterative
  LOESS-Anpassung mit Ausreißer-Downweighting (robuste STL-Variante,
  Cleveland u. a. 1990). Verhindert, dass einzelne Anomalien Trend und
  Saisonalität zu sich hin verbiegen und sich so „wegerklären".
- **Einmaliger Fit auf der gesamten 4-Jahres-Reihe.** STL ist ein
  lokaler LOESS-Smoother; Leakage über Train/Val/Test-Grenzen hinweg
  ist gering und wird zugunsten von Reproduzierbarkeit und Kontinuität
  in Kauf genommen. Diese Designentscheidung ist hier explizit
  dokumentiert.
- **Eine unabhängige Dekomposition pro Zähler** (univariates
  Verfahren).

### 4.3 NaN-Behandlung

`statsmodels.STL` akzeptiert keine fehlenden Werte. Vorgehen:

1. Eingangsserien werden vor dem Fit linear interpoliert (auch über
   lange NaN-Strecken hinweg).
2. Nach der Dekomposition wird das Residuum an den ursprünglich
   fehlenden Positionen wieder auf NaN gesetzt, damit nachgelagerte
   Modelle nicht auf erfundenen Werten trainieren.
3. Trend- und Saisonalitätskomponenten an Lücken-Positionen bleiben
   als gefittete Werte erhalten; sie dürfen als geglättete
   Extrapolation für die deskriptive Analyse verwendet werden,
   jedoch mit dem Hinweis, dass sie in der Nähe langer Lücken
   (insbesondere wmz_3) unsicher sind.

### 4.4 Output und Sanity-Kennzahlen

`outputs/<dataset>/parquet/stage3_stl.parquet` — **35 064 × 12**:

| Spalten je Zähler | Bedeutung |
|---|---|
| `wmz_N_trend` | LOESS-Trend; langsame Komponente |
| `wmz_N_seasonal_24h` | tägliche Saisonalität |
| `wmz_N_seasonal_168h` | wöchentliche Saisonalität |
| `wmz_N_residual` | Rest (NaN an Original-Lücken-Positionen) |

Anteil der Varianz pro Komponente (grobe Sanity-Check-Kennzahl,
nicht additiv da Komponenten korreliert sind):

| Zähler | Trend | Saison 24h | Saison 168h | Residuum |
|---|---|---|---|---|
| wmz_1 | 0.399 | 0.140 | 0.106 | 0.345 |
| wmz_2 | 0.125 | 0.643 | 0.260 | 0.213 |
| wmz_3 | 0.533 | 0.211 | 0.104 | 0.234 |

Residuum-Statistiken (NaN an Lücken-Positionen restauriert):

| Spalte | Mittelwert | StdAbw | Min | Max |
|---|---|---|---|---|
| wmz_1_residual | −0,02 | 0,83 | −10,9 | +17,9 |
| wmz_2_residual | +0,59 | 5,58 | −37,6 | +58,1 |
| wmz_3_residual | +4,87 | 57,19 | −353,7 | +1200,3 |

### 4.5 Visualisierungen für die Mustererkennung

Erzeugt durch [src/tools/stage3_explore.py](../src/tools/stage3_explore.py),
Ablage in `outputs/<dataset>/figures/`:

| Plot | Inhalt |
|---|---|
| `stl_zoom_wmz_{1,2,3}.png` | 4-Wochen-Zoom: Original, Trend, Saison 24h, Saison 168h, Residuum übereinander |
| `stl_trend_alle.png` | Trend über die vollen 4 Jahre, alle drei Zähler — zeigt Jahresgang und langfristige Verschiebungen |
| `stl_tagesprofil.png` | Mittelwert der Saison-24h-Komponente pro Stunde des Tages |
| `stl_wochenprofil.png` | Mittelwert der Saison-168h-Komponente pro Wochentag |
| `stl_residual_hist.png` | Verteilung der Residuen (log-y) — der Eingang der AD-Modelle |

---

## 5. Empirische Befunde nach Stage 2

Generiert über das Exploration-Skript
[src/tools/stage2_explore.py](../src/tools/stage2_explore.py); Abbildungen in
[outputs/figures/](../outputs/gebaeude_a/figures/).

### 5.1 Physikalische Zuordnung der drei Zähler

| | wmz_1 | wmz_2 | wmz_3 |
|---|---|---|---|
| Tages-Mittel | konstant 9–11 kW | Peak 18–20 morgens, 5 nachts | Peak 150 morgens, 55 nachts |
| Sommer (Jun–Aug) | 9 kW | 5 kW | **3–4 kW** |
| Winter (Dez–Feb) | 11 kW | 15–16 kW | **195–225 kW** |
| Korrelation zur Außentemperatur | r = −0.34 | r = −0.37 | **r = −0.67** |

Die datengetriebenen Profile decken sich mit den realen
Versorgungsfunktionen der drei Wärmeübertrager (Betreiberauskunft des Wärmeversorgers;
Achtung: die Daten-Spalten `wmz_1/2/3` sind gegenüber der Nummerierung des Wärmeversorgers
„Wärmeübertrager 1/2/3" vertauscht, s. `stage1_load.py`):

- **wmz_3** = **Raumheizung** (Laborgebäude + Fußbodenheizung Küche):
  50-fach höherer Mittelwert im Winter, starke negative Temperaturkorrelation,
  Sommerverbrauch ≈ 0 — passt zur Heizfunktion.
- **wmz_1** = **Trinkwarmwasser der Allgemeinbereiche**: schwach strukturiert,
  von Zirkulations-/Grundlast statt scharfen Zapfspitzen geprägt.
- **wmz_2** = **Trinkwarmwasser + dynamische Küchenheizung**:
  nutzungs-getriebener Tagespeak (Essenszeiten/Werktagsbetrieb der Küche),
  Sommer reduziert aber nicht auf 0.

### 5.2 Jahresverbrauch

| Jahr | wmz_1 [MWh] | wmz_2 [MWh] | wmz_3 [MWh] |
|---|---|---|---|
| 2019 (Nov–Dez) | 11,0 | 14,6 | 32,5 |
| 2020 | 80,7 | 75,8 | 801,8 |
| 2021 | 88,5 | 96,6 | 888,0 |
| 2022 | 83,6 | 96,9 | 726,1 |
| 2023 (bis 19.11.) | 78,7 | 73,0 | 444,3 |

Der Einbruch bei wmz_3 in 2023 ist kein realer Verbrauchsrückgang,
sondern Folge von Datenausfällen (8,2 % NaN-Stunden plus
abgeschnittenes Jahresende). Jahresvergleiche müssen auf
Verfügbarkeit normiert werden.

### 5.3 Datenqualität verschlechtert sich über die Zeit

Stunden mit ≥1 minuten-level Glitch pro Jahr:

| Jahr | wmz_1 | wmz_2 | wmz_3 |
|---|---|---|---|
| 2019 | 51 | 60 | 807 |
| 2020 | 446 | 496 | 755 |
| 2021 | 494 | 423 | 1 011 |
| 2022 | 1 187 | 415 | 1 199 |
| 2023 | 3 058 | 344 | 3 566 |

wmz_1 und wmz_3 zeigen einen **Faktor ~6 Anstieg** von 2019 zu 2023.
wmz_2 bleibt stabil. Die Hardware-Qualität ist über den
Beobachtungszeitraum nicht konstant — eine eigenständig erwähnenswerte
Beobachtung für eine Studie zur Anomaliedetektion unter realistischen
Bedingungen.

### 5.4 Methoden-zentrierte Datenexploration (`src/exploration/`)

Ergänzend zu den stage-spezifischen Plots in `src/tools/` enthält
[src/exploration/](../src/exploration/) **methoden-zentrierte**
Auswertungen, die nicht einer einzelnen Pipeline-Stage zugeordnet sind,
sondern allgemein die Daten-Eigenschaften herausarbeiten und damit
Modellierungs- bzw. Feature-Entscheidungen empirisch stützen.

| Skript | Methode | Quelle |
|---|---|---|
| [temperature_scatter.py](../src/exploration/temperature_scatter.py) | Scatter Verbrauch × Aussentemperatur mit LOWESS-Glättung (`frac=0.3`, robust iteriert) | Cleveland 1979 (LOWESS); Hammarsten 1987 (Energy-Signature-Knick als interpretativer Rahmen) |
| [hour_weekday_heatmap.py](../src/exploration/hour_weekday_heatmap.py) | 24 × 7-Heatmap mittlerer kW pro (Stunde × Wochentag), je WMZ | Tukey 1977, EDA |
| [autocorrelation.py](../src/exploration/autocorrelation.py) | ACF und PACF über 168 Lags (1 Woche), mit roten Hilfslinien bei Lag 24 und Lag 168 | Box und Jenkins 1976 |
| [load_duration_curve.py](../src/exploration/load_duration_curve.py) | Jahresdauerlinie (absteigend sortierte Last vs. Stunden-Anteil) je WMZ; Grundlast-/Spitzenlast-Anteil | Verbruggen 1980 (Methode); Frederiksen und Werner 2013 (Fernwärme-Lehrbuch, aktuell) |
| [monthly_boxplot.py](../src/exploration/monthly_boxplot.py) | Monatliche kW-Verteilung je WMZ als Boxplot-Reihe; Drift-/Strukturbruch-Diagnose (EnSikuMaV-Marker) | Tukey 1977 |
| [temperature_crosscorrelation.py](../src/exploration/temperature_crosscorrelation.py) | Kreuzkorrelation kW ↔ Aussentemperatur über Lags 0–24 h; thermische Trägheit | Box und Jenkins 1976 |
| [stl_strength.py](../src/exploration/stl_strength.py) | numerische Trend-/Saison-Stärke `F = max(0, 1 − Var(R)/Var(K+R))` je WMZ aus der Stage-3-MSTL-Zerlegung (CSV + Plot) | Wang, Smith und Hyndman 2006 |
| [mutual_information.py](../src/exploration/mutual_information.py) | Mutual Information vs. Pearson-r für kW × Aussentemperatur; nicht-lineare Abhängigkeit (CSV + Plot) | Kraskov, Stögbauer und Grassberger 2004 |
| [cross_meter_correlation.py](../src/exploration/cross_meter_correlation.py) | Paarweise Pearson-/Spearman-Korrelation der 3 WMZ je Signalebene (Roh-kW / MSTL-Trend / Residuum); Heatmap + CSV. Stützt die Architektur-Entscheidung lokal vs. global (§7.11) | Box und Jenkins 1976 (Kreuzkorrelation); Montero-Manso und Hyndman 2021 (global vs. lokal) |

**Was diese drei Plots methodisch liefern:**

* Der **LOWESS-Scatter** zeigt den nicht-linearen Knick der
  Heizleistung an der **Heizgrenztemperatur**: unterhalb steigt der
  Bedarf näherungsweise linear mit fallender Temperatur, oberhalb
  bleibt er nahe null. Das ist die empirische Basis für die
  Behandlung der Temperatur als nicht-linearer Prädiktor (bzw. für
  die Wahl der Modelle, die solche Knicke erlernen können — IForest
  und LSTM-AE tun das, Z-Score nicht). Die Aufteilung der Punkte
  nach Heiz- vs. Sommer-Saison (rote/blaue Punkte) macht zusätzlich
  sichtbar, dass die Heizgrenze nicht nur eine Funktion der Temperatur,
  sondern auch der Jahreszeit ist (Sommer-Stillstand auch bei kühleren
  Nächten).
* Die **Wochenprofil-Heatmap** ist die kompakteste Form, das in
  Stage 3 angesetzte 168-h-Wochenprofil empirisch zu rechtfertigen —
  WMZ mit Werktag/Wochenend-Unterschied liefern klar getrennte
  Spaltenstrukturen, WMZ ohne (wmz_3 ist hier auffällig) sind
  homogener.
* **ACF/PACF** auf 168 Lags zeigen scharfe Peaks bei Lag 24 und Lag
  168, was die Wahl der MSTL-Perioden `[24, 168]` empirisch
  legitimiert. Bei wmz_3 sind die Peaks deutlich schwächer — passend
  zu dem in §5.1 / §5.3 dokumentierten Glitch-Anteil dieses Zählers.

Fünf weitere Skripte vertiefen die Charakterisierung (Backlog 3a–3e):

* Die **Jahresdauerlinie** ordnet jeden Zähler in das Schema
  Grundlast/Spitzenlast ein: ein steiler linker Rand mit flachem
  Plateau zeigt wetterabhängige Spitzenlast (Raumheizung), eine flache
  Kurve eine konstante Grundlast (Warmwasser-/Prozesslast) — stützt die
  WMZ-Interpretation aus §5.1 quantitativ.
* Der **Monats-Boxplot** macht Langfrist-Drift und Strukturbrüche
  sichtbar (EnSikuMaV-Marker ab Sep 2022) und liefert über den
  Ausreißer-Anteil je Monat einen rohen Glitch-Indikator; der robuste
  Boxplot ist gegenüber Einzel-Extremwerten unempfindlich.
* Die **Kreuzkorrelation** über Lags 0–24 h quantifiziert die
  thermische Trägheit (Lag des betragsmaximalen r) und prüft, ob
  lagged-Temperatur-Features Zusatzinformation trügen.
* Die **STL-Strength-Maße** ersetzen das qualitative Ablesen der
  MSTL-Plots durch eine zwischen den Zählern vergleichbare Kennzahl
  (`F_Trend`, `F_Saison`). Im Datensatz: stärkster Trend bei wmz_3
  (`F_Trend` ≈ 0,82, passend zur ausgeprägten Temperaturabhängigkeit),
  schwächste Tages-/Wochen-Saisonalität bei wmz_1 (`F_Saison_24h` ≈ 0,30).
* Die **Mutual Information** erfasst — anders als Pearson-r — auch die
  nicht-lineare Temperaturabhängigkeit; eine Konstellation „hohe MI bei
  niedrigem |r|" rechtfertigt empirisch, Temperatur als Feature für
  Modelle bereitzustellen, die Nicht-Linearitäten lernen können
  (IForest, LSTM-AE).

Die neun Skripte sind eigenständig lauffähig (`python
src\exploration\<skript>.py --dataset <name>`) und produzieren PNGs
in `outputs/<dataset>/figures/`. Sie sind kein Bestandteil der
Detektions-Pipeline und werden in der Stage-Reihenfolge nicht
benötigt; sie dienen ausschließlich der Thesis-Diskussion und der
empirischen Begründung der getroffenen Methodik-Entscheidungen.

### 5.5 Implikationen für die Folge-Stages

- **Stage 3 (Features):** Temperatur ist primär für wmz_3 relevant.
  Univariate Modelle pro Zähler werden vermutlich unterschiedlich
  von Wetter-Features profitieren.
- **Stage 4 (Split):** Der Test-Bereich (Mai–Nov 2023) liegt im
  Jahr mit der schlechtesten Datenqualität. Der Test ist damit
  unbeabsichtigt aber wertvoll „realistischer" als der Trainings-
  bereich; das gehört explizit in die Diskussion.
- **Stage 6 (Modelle):** Die Sommer-Nullen von wmz_3 sind
  legitimer Normalbetrieb. Distanzbasierte Verfahren (LOF,
  IsoForest) ohne kontextuelles Feature werden den
  Heizperioden-Wechsel ansonsten als Anomalie sehen.
- **Stage 8 (Anomalie-Injektion):** Synthetische
  „Drop ×0"-Anomalien sind im Sommer für wmz_3 nicht von der
  Grundlinie unterscheidbar (Normalwert ist bereits 0). Injektion
  sollte auf Heizperioden begrenzt oder hierfür entsprechend
  bewertet werden.

---

## 6. Pipeline-Architektur (übergreifend)

Die Gesamt-Pipeline ist auf eine vergleichende Auswertung dreier
parallel laufender Detektions-Varianten ausgelegt. Diese Architektur
ist die methodische Grundlage für die Untersuchung, welcher Grad
an Vorverarbeitung optimal mit den unterschiedlichen Anomalie-Typen
zusammenspielt (vgl. Hypothese H3, MSTL-Mehrwert).

### 6.0 Gesamt-Pipeline: Datenfluss und Begründungs-Faden

Dieser Abschnitt erklärt **didaktisch, was in welcher Reihenfolge
passiert und warum** — die einzelnen Stages sind in §§ 2–4 und 7
ausführlich beschrieben, hier steht der rote Faden zwischen ihnen.

#### 6.0.1 Datenfluss-Diagramm

```
       Roh-CSVs (5x WMZ je 1 min + 1x Open-Meteo, stuendlich)
                              |
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 1  - Laden + Mergen + 4-Jahres-Clipping            │
   │           src/stage1_load.py                             │
   └──────────────────────────────────────────────────────────┘
                              |
                              v  ~2,1 Mio Minuten-Zeilen
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 2  - Dual-Channel-Fehlerfilter (Kategorie-A-Flags) │
   │            + Stunden-Aggregation                         │
   │           src/stage2_preprocess.py                       │
   └──────────────────────────────────────────────────────────┘
                              |
                              v  35 064 Stunden (4 Jahre)
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 3  - MSTL-Dekomposition (Trend + 24h + 168h        │
   │            Saison + Residuum)                            │
   │           src/stage3_stl.py                              │
   └──────────────────────────────────────────────────────────┘
                              |
            ┌─────────────────┼──────────────────┐
            v                 v                  v
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │ Stage 4  -     │ │ Stage 4  -     │ │ Stage 3-       │
   │ Features raw   │ │ Features resid │ │ Trend direkt   │
   │ src/stage4_*   │ │ src/stage4_*   │ │ (keine Stage 4)│
   └────────────────┘ └────────────────┘ └────────────────┘
            |                 |                  |
            └─────────────────┼──────────────────┘
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 5  - Train/Val/Test-Split (zeitlich strikt,        │
   │            kein Shuffling)                               │
   │           src/stage5_split.py                            │
   └──────────────────────────────────────────────────────────┘
                              |
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 6  - Normalisierung (Z-Score, ausschliesslich auf  │
   │            Train-Statistik gefittet → kein Leakage)      │
   │           src/stage6_normalize.py                        │
   └──────────────────────────────────────────────────────────┘
                              |
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 7  - Modell-Training (Defaults; primaer Smoke-     │
   │            Test des Pipeline-Pfads)                      │
   │           src/stage7_train.py                            │
   └──────────────────────────────────────────────────────────┘
                              |
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 8  - HPO: Validation-Injektion (VAL_SEED=8) +      │
   │            Selektion nach ROC-AUC/PR-AUC/F1              │
   │           src/stage8_hpo.py                              │
   └──────────────────────────────────────────────────────────┘
                              |
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 9  - Test-Injektion (TEST_SEED=99) + Ground Truth  │
   │           src/stage9_inject.py                           │
   └──────────────────────────────────────────────────────────┘
                              |
                              v
   ┌──────────────────────────────────────────────────────────┐
   │ Stage 10 - Test-Evaluation (eingefrorener Maßstab)       │
   │            + 10b Qualitativ + 10c EnSikuMaV-Validierung  │
   │           src/stage10_evaluate.py                        │
   └──────────────────────────────────────────────────────────┘

   Querschnitts-Werkzeuge (eigene §-Blöcke):
   * src/exploration/   Datenexploration (§ 5.4)
   * src/tools/data_sweep.py            Lernkurven (§ 7.8)
   * src/tools/seed_variance.py         HPO-Diagnose (§ 7.12)
   * src/tools/score_distributions.py   Modell-Diagnose (§ 7.12)
   * src/tools/lag_ablation.py          Lag-Feature-Ablation (§ 7.1)
   * src/result_io.py                   Nicht-destruktives Schreiben
                                        in Stage 8 + Stage 10
```

#### 6.0.2 Pro Stage: was passiert und *warum*

| Stage | Was | Warum gerade so |
|---|---|---|
| **1 Laden** | 5 WMZ-CSVs + Open-Meteo zusammenführen, auf das 4-Jahres-Fenster `[2019-11-19, 2023-11-19)` clippen | Konsistenter Zeitraum, Sommerzeit-Übergänge zweimal komplett enthalten, exklusives Ende identisch zu Stage 5; ein einziger Wahrheits-Block für alle Folge-Stages. |
| **2 Vorverarbeitung** | Sechs regelbasierte Fehler-Detektoren auf MWh **und** kW, cross-channel-Plateau-Detektion, dann Bereinigung und Stunden-Aggregation | Trennt **bekannte Datenfehler** (Kategorie A) sauber von noch unbekannten Anomalien (Kategorie B), bevor irgendein Modell die Daten sieht. Die Fehler-Flags werden **bewusst nicht** als Modell-Input verwendet (sie wären in einer Produktiv-Inferenz nicht verfügbar), sondern nur als Trainings-Filter (§ 6.4) und Eval-Ground-Truth. |
| **3 MSTL** | Multi-Saisonale STL-Dekomposition `[24, 168]` mit `robust=True` | Liefert drei für die Anomaliedetektion fundamental unterschiedliche Signale: **Trend** (langsam, für Drift/Strukturbruch), **Saison** (Periodik, wird abgetrennt), **Residuum** (stationäres Rauschen, ideale Eingabe für distance-/density-basierte Detektoren). `robust=True` verhindert, dass einzelne Anomalien den Trend zu sich heranziehen. |
| **4 Feature Engineering** | Pro Variante eine Feature-Tabelle: raw (volle Features), residual (minimal-Kontext) | Macht aus den Stage-3-Outputs **vergleichbare Modell-Eingaben**. Die Asymmetrie raw=multivariate / residual=minimal ist die methodische Antwort auf die Hypothese H3 („nimmt STL den Modellen Arbeit ab?"). |
| **5 Split** | Zeitlich strikt: Train = bis 2022-11-01, Val = bis 2023-05-01, Test = bis 2023-11-19 | **Kein Shuffling**, kein k-fold. Begründung Bergmeir und Benítez 2012: zufällige Splits in Zeitreihen produzieren Leakage durch Auto-Korrelation. Test-Bereich fällt zudem in das *schlechteste* Datenjahr (siehe § 5.3) — realistischer als Best-Case-Train. |
| **6 Normalisierung** | StandardScaler pro Variante, gefittet **nur** auf Train-Zeilen | Distanz-/Dichte-/Baum-basierte Modelle sind skalen-sensitiv. Fit auf Train + transform auf Val/Test ist die einzige Reihenfolge ohne Test-Leakage (Pedregosa u. a. 2011, Standard-ML-Praxis). Mean/Std werden persistiert für Produktiv-Inferenz. |
| **7 Training** | Fit aller Modelle pro `(Variante × WMZ)` mit Default-HPs | Pipeline-Smoke-Test + Persistenz von `.pkl`-Dateien für Ad-hoc-Inferenz. Die Test-relevanten Modelle entstehen **erst in Stage 10** mit den HPs aus Stage 8 frisch — Stage 7 ist also bewusst redundant zur Stage-10-Eval, dient aber dem schnellen Pipeline-Check. |
| **8 HPO** | Synthetische Injektion **in die Validation-Menge** (`VAL_SEED=8`), Selektion nach ROC-AUC/PR-AUC/F1; Random-Search bzw. gridwh für LSTM-AE, vollständiger Grid für die Klassiker | Unsere Detektoren sind **unüberwacht** — auf realen Daten fehlt das Label-Signal. Lösung: kontrollierte Injektion mit bekannter Wahrheit erzeugt einen *messbaren* Optimierungs-Zielwert. **Eigener Val-Seed** (≠ Test-Seed 99) verhindert HP-Overfitting auf die exakten Test-Anomalien. |
| **9 Test-Injektion** | Identische Injektions-Maschinerie wie Stage 8, aber auf das Test-Fenster und mit `TEST_SEED=99`; volle (35 064 h) Reihe gespeichert, nur Test-Stunden gestört | Trennt sauber zwischen *Tuning-Daten* (Stage 8) und *Bewertungs-Daten* (Stage 9), damit die in § 7.7 beschriebene Objektivität trägt. Volle Reihe persistiert, weil sequenzielle Modelle (LSTM-AE) den sauberen Vor-Kontext für ihre Fenster brauchen. |
| **10 Evaluation** | Test-Inferenz mit Stage-8-HPs, point-adjusted P/R/F1 + ROC-/PR-AUC, stratifiziert nach Anomalietyp und Intensität | Stage 10 ist der **eingefrorene Bewertungs-Block** (§ 7.7): über alle Experimente (klassisch vs. LSTM, lokal vs. global, jedes HPO-Kriterium, jede Daten-Sweep-Stufe) bleibt das Lineal identisch — nur Kandidaten und Trainings-Input variieren. Alles andere wäre kein A/B. |

#### 6.0.3 Architektur-Entscheidungen, die sich durch die Pipeline ziehen

Diese sechs Entscheidungen lassen sich nicht einer einzelnen Stage
zuordnen, sondern sind **Quer­schnitts-Architektur**. Sie zusammen
machen die Pipeline zu einer Pipeline und nicht zu einer Aneinander­
reihung unabhängiger Schritte.

1. **Pro-WMZ-Trennung statt globalem Modell.** `iter_training_jobs()`
   in `src/models/registry.py` erzeugt für jede `(Variante × WMZ ×
   Modell)`-Kombination einen eigenen Job; `model_features()`
   schließt die jeweils anderen Zähler **explizit aus** (Anti-
   Cross-Leakage). Begründung: die drei Zähler haben um Faktor ~10
   unterschiedliche kW-Niveaus (Stage-2-Sanity, § 3.4), unterschiedliche
   STL-Saisonalitätsstärke (`F_Saison` 0,30–0,62) und unterschiedliche
   Fehlerraten (5–52 % geflaggte Stunden). Ein einziges globales Modell
   wäre an dieser Heterogenität gescheitert. Globales Training ist
   **dokumentierter Ausblick** (§ 7.11, Entscheidung gegen Umsetzung
   2026-06-04, Korrelations-Befund: WMZ-Residuen ≈ unkorreliert).
2. **Drei parallele Varianten statt einer „besten".** Anomalietypen
   und Eingabe-Repräsentationen sind asymmetrisch verkoppelt (§ 6.3):
   stationäre Anomalien sind im Residuum gut sichtbar, nicht-
   stationäre nur im Trend. Eine einzige „beste" Variante hätte
   per Konstruktion eine ganze Klasse von Anomalien verfehlt.
3. **Saubere Train/Val/Test-Trennung mit drei Seeds.** Stage 5 fixiert
   die Splits zeitlich; Stage 6 fittet Scaler nur auf Train; Stage 8
   injiziert in Val mit `VAL_SEED=8`; Stage 9 injiziert in Test mit
   `TEST_SEED=99`. **Drei separate, deterministische Randomness-
   Quellen** — kein einziger Bewertungsschritt sieht die Wahrheit
   des nächsten.
4. **Eingefrorener Bewertungs-Maßstab (§ 7.7).** Was zwischen den
   Experimenten variieren *darf*: Suchstrategie, HPs, Datenmenge,
   Modell-Familie. Was **nicht** variieren darf: Test-Set, Injektion,
   Ground Truth, Metriken, Hardware/Präzision. Sehili und Zhang 2023
   zeigen, wie schnell Deep-AD-Vergleiche durch unbemerkt mitwandernde
   Bewertungsbausteine wertlos werden — die strikte Einfrierung ist
   die methodische Antwort darauf.
5. **Nicht-destruktives Schreiben (`src/result_io.py`).** Stage 8 und
   Stage 10 schreiben `best_hparams.json` bzw. `stage10_metrics.csv`
   seit dem `result_io`-Umbau als **Merge in Bestehendes**, nicht als
   Komplett-Überschreibung. Ein Teilmenge-Lauf (z. B. `--models
   lstm_ae` für ein methodisches Folge-Experiment) ersetzt nur die
   in diesem Lauf bewerteten `(variant, wmz, model)`-Zeilen und
   behält den Rest. Erst ein expliziter `--fresh`-Flag erzwingt
   Komplett-Neuschrieb. Konsequenz: die Pipeline überlebt
   iterative methodische Experimente ohne kumulative Backup-Choreografie
   (das frühere manuelle Snapshot/Merge-Tool `merge_results.py` wurde
   damit obsolet und am 2026-07-15 entfernt).
6. **Diagnose-Werkzeuge als separate `src/tools/`-Skripte.** Was nicht
   Bestandteil der Pipeline ist, sondern *Postprocessing zur Beantwortung
   methodischer Folgefragen* (Datenmengen-Sensitivität § 7.8,
   Seed-Varianz § 7.12, Score-Verteilungs-Diagnose § 7.12), liegt
   bewusst außerhalb der `stage{1..10}_*.py`-Pipeline. So bleibt die
   Pipeline reproduzierbar deterministisch, während die Diagnose-Tools
   die Pipeline-Outputs aus verschiedenen analytischen Perspektiven
   beleuchten.

#### 6.0.4 Wann wird die Pipeline „komplett" durchlaufen?

Drei Anlässe für einen vollen Stages-1-bis-10-Lauf:

- **Initial-Lauf auf einem neuen Datensatz** (`--dataset <name>` an
  jeder Stage): die Pipeline ist über `--dataset` parametrisiert und
  läuft ohne Code-Änderung; das ist auch der Mechanismus für Stage 11
  (Robustheits-Evaluation, § 7.9).
- **Methodische Änderung an einer frühen Stage** (z. B. neuer
  Fehler-Detektor in Stage 2 oder anderes Feature-Set in Stage 4):
  alle nachgelagerten Stages **müssen** neu laufen, weil sie auf den
  Parquets der vorigen Stage aufsetzen.
- **Modell-/HPO-Änderung** (`src/models/*` oder Stage 8): Stage 7,
  8, 10 reichen — die Stages 1–6 sind deterministisch und ändern
  sich nicht. Diese Form des Teil-Reruns ist seit dem `result_io`-
  Umbau gefahrlos möglich (s. Punkt 5 oben).

### 6.1 Anomalie-Kategorien

Drei strukturell verschiedene Anomalie-Klassen, jeweils mit eigener
Ursache, eigener Erkennungs-Schiene und eigener praktischer
Konsequenz. Die Unterscheidung baut auf den etablierten Taxonomien der
Anomaliedetektion auf (Punkt-, kontextuelle, kollektive Anomalien bei
Chandola u. a. 2009; verhaltensbasierte Zeitreihen-Taxonomie mit Trend-
und Saison-Outliern bei Lai u. a. 2021):

| Kategorie | Beispiele | Detektor-Schiene |
|---|---|---|
| **A — Datenfehler** | Sensor-Glitches, Counter-Reset, Sensor-Plateau, Datenlücken | Regelbasiert in Stage 2 (Fehlerfilter); Flags als Eval-Ground-Truth |
| **B — System-Anomalien** | Spike, Drop, Plateau, Leckage, Drift, Strukturbruch (in Stage 9 synthetisch injiziert) | Variante A und B (Z-Score, LOF, IF, LSTM-AE) für stationäre Typen; Variante C (PELT) für nicht-stationäre |
| **C — Operative/Regulatorische Änderungen** | EnSikuMaV September 2022, COVID-Lockdowns | Variante C (PELT auf Trend) als labelfreie Realwelt-Validierung |

Konsequenz für die Auswertung: ein Modell-Alarm wird **stratifiziert
nach Kategorie** ausgewertet — eine Detektion ist „True Positive"
nur in Bezug auf die Kategorie, gegen die gerade evaluiert wird.

### 6.2 Drei Detektions-Varianten

| Variante | Eingang | Feature-Anzahl | Modelle | Zielanomalien |
|---|---|---|---|---|
| **A — Rohdaten** | stündliche kW-Mittelwerte (+ Time-/Wetter-Features) | ~12 | Z-Score, LOF, IF, LSTM-AE, Constancy (plateau-spezifisch) | Spike, Drop, Plateau, Leckage |
| **B — MSTL-Residuum** | stündliches Residuum (+ minimaler Kontext) | ~5 | Z-Score, LOF, IF, LSTM-AE | Spike, Drop, Plateau (gleiche Anomalien, aber auf deseasonalisiertem Signal) |
| **C — Trend-Schiene** | MSTL-Trend | 1 (univariat) | PELT (klassisch), LSTM-AE (Vergleich) | Drift, Strukturbruch |

Varianten A und B sind eine **Ablation** der STL-Vorverarbeitung —
gleiche Modelle, unterschiedliche Eingänge — und beantworten die
Hypothese, ob STL methodischen Mehrwert bringt. Variante C ist
strukturell komplementär: Anomalietypen, die A und B per Definition
nicht finden können (weil sie vom Trend absorbiert werden), sind
genau die, die C explizit sucht.

### 6.3 Stationäre vs. nicht-stationäre Anomalien

Die Aufteilung A/B vs. C entspricht einer Aufteilung nach der
**Stationaritäts-Eigenschaft der Anomalie**. Maßgeblich ist, ob die Anomalie
die *Verteilung bzw. das Normalniveau* der Zeitreihe **dauerhaft verändert**:

- **Stationäre Anomalien** (Spike, Drop, Plateau, kurze Leckage): *zeitlich
  begrenzte* Auslenkung; nach Ende der Anomalie kehrt das Signal in seine
  **ursprüngliche Verteilung** zurück — das Normalniveau bleibt unverändert.
  Der STL-**Trend reagiert kaum**, das **Residuum bleibt informativ** → A/B.
  *Hinweis:* „stationär" bezieht sich auf die Rückkehr zum Niveau; ein Plateau
  ist davon unbenommen eine **kollektive** Anomalie (anomal ist die Konstanz,
  nicht ein Ausreißerwert) und wird daher vom Constancy-Detektor adressiert
  (§ 6.6 / Detektor-Bericht), nicht von den Punktverfahren.
- **Nicht-stationäre Anomalien** (Drift, Strukturbruch, dauerhafte Leckage,
  EnSikuMaV): *persistente oder graduelle* Verschiebung; das **Normalniveau
  selbst ändert sich dauerhaft** (Bruch in den Verteilungsparametern). Der
  STL-**Trend absorbiert** die Verschiebung, das **Residuum verliert das
  Signal** → Schiene C (PELT auf dem Trend).

| Eigenschaft | **Stationär** (A/B) | **Nicht-stationär** (C) |
|---|---|---|
| Wirkung aufs Normalniveau | kehrt zurück | **bleibt dauerhaft verschoben** |
| Zeitlichkeit | begrenzt (Stunden–wenige Tage) | persistent / graduell (Wochen) |
| Typen | Spike, Drop, Plateau, kurze Leckage | Drift, Strukturbruch |
| Informatives Signal | **Residuum** (Trend reagiert kaum) | **Trend** (Residuum verliert es) |
| Detektoren | Z-Score, LOF, IF, LSTM-AE, **Constancy** (Plateau) | **PELT**, LSTM-AE |
| Ground-Truth-Spalte | `gt_stat_<wmz>` | `gt_nonstat_<wmz>` |

Daraus folgt zwingend, dass Varianten A/B und Variante C **disjunkte**
Anomalietypen adressieren — keine Schiene ist redundant; die Aufteilung der
Ground-Truth (`gt_stat` / `gt_nonstat`) folgt exakt dieser Achse. Konkrete
Intensitäten/Dauern siehe Injektions-Katalog (Stage 9, § 7.6).

**Vollständige Zuordnung: Anomalieart → Ausprägung → Detektor.** Die folgende
Tabelle verbindet die allgemeine AD-Taxonomie (Punkt-/kontextuelle/kollektive
Anomalien; Chandola u. a. 2009) mit der Stationaritäts-Achse, der konkreten
Ausprägung im Wärme-/Leistungssignal (kW) und dem zuständigen Detektor. Der
empirische Recall stammt aus dem jeweiligen Headline-Job (Ergebnisbericht
§ 11.4 / § 11.4.1).

| AD-Klasse (Chandola) | Stationarität | Typ | Ausprägung in Leistung/Energie (kW) | Variante | Primärer Detektor | Empirie (Recall, bester Job) |
|---|---|---|---|---|---|---|
| **Punkt** | stationär | **Spike** | kurzer Leistungs-Peak (× 3–10 Median), ~1 h; Rückkehr zur Verteilung | A, B | Z-Score, IForest, LOF | **1,00** (alle) |
| **Punkt** | stationär | **Drop** | kurzer Einbruch unter das erwartete Niveau, ~1 h | A, B | Z-Score, IForest | Z 1,00 · IF 0,67 |
| **Kontextuell / kollektiv** | stationär | **Leckage** | dauerhafter **additiver Mehrverbrauch** (Leck/Bypass), erhöhtes Niveau über Stunden–Tage, dann zurück | A, B | IForest, LOF | IF **1,00** · LOF 0,88 |
| **Kollektiv** | stationär | **Plateau** | Leistung **eingefroren/konstant trotz erwarteter Aktivität** (klemmendes Ventil, fehlende Modulation); Einzelwerte plausibel, *die Konstanz* ist anomal | A | **Constancy** (Varianz-Einbruch) | **1,00** (wmz_1/2) · Standard-Detektoren **0,00** |
| **Change-Point** | **nicht-stationär** | **Drift** | langsames **Wegdriften des Normalniveaus** (Verkalkung, ΔT-Verschlechterung, Effizienzverlust) | C (Trend) | **PELT** | 0,25 → **0,75** (mit Intensität) · LSTM-AE 0,63 |
| **Change-Point** | **nicht-stationär** | **Strukturbruch** | **abrupter, dauerhafter Niveauwechsel** (EnSikuMaV-Regulierung, Nutzungsänderung, Anlagentausch) | C (Trend) | **PELT** | 0,33 · LSTM-AE 0,67 |

Lesehilfe: **stationär** (Punkt + kollektiv) → das *Residuum* ist informativ →
A/B; **nicht-stationär** (Change-Point) → der *Trend* ist informativ → C.
Sonderfall **Plateau**: kollektive Anomalie ohne Punktausreißer und ohne
Drift → Recall 0 bei allen fünf Standard-Detektoren (inkl. LSTM-AE, der eine
konstante Sequenz fehlerfrei rekonstruiert); erst der `ConstancyDetector` (§ 6.6)
schließt die Lücke. Verfahrensfamilien: LOF dichtebasiert (Breunig u. a. 2000),
IForest ensemble/isolation (Liu u. a. 2008), Constancy Varianz-Change-Point
(Inclán und Tiao 1994; Killick u. a. 2012), PELT Change-Point (Killick u. a.
2012), LSTM-AE Rekonstruktion (Malhotra u. a. 2016).

### 6.4 Trainings-Filterung

Die in Stage 2 produzierten **Fehler-Flags sind kein Modell-Input**
(in einer Produktiv-Deployment-Situation wären sie zum
Detektionszeitpunkt nicht verfügbar), sondern erfüllen zwei separate
Funktionen:

1. **Trainings-Daten-Filterung** (Stage 6): geflaggte Zeitstempel
   werden aus dem Trainings-Set ausgeschlossen, damit Modelle nicht
   versehentlich Daten-Fehler als Normal-Muster lernen.
2. **Evaluations-Ground-Truth** (Stage 10): Detektionen im Test-Set
   werden gegen die Flag-Spalten geprüft, um die Kategorie A einer
   Detektion bestimmen zu können.

### 6.5 Verortung der Verfahren: KI (weit/eng) und Muster- vs. Anomalieerkennung

Die Pipeline kombiniert bewusst klassische und lernende Verfahren. Da der
Arbeitstitel „KI-basierte Methoden zur Muster- und Anomalieerkennung"
nennt, wird hier präzisiert, welcher Baustein in welchem Sinn „KI" ist und
wo Mustererkennung tatsächlich stattfindet.

**KI im weiten vs. engen Sinn** (Schachtelung KI ⊃ ML ⊃ Deep Learning):

| Baustein | Stage | Einordnung | KI? |
|---|---|---|---|
| MSTL-Dekomposition | 3 | klassische Statistik (LOESS) | nein |
| Z-Score | 7 | statistische Baseline | nein |
| PELT | 7 | klassische Changepoint-Statistik | nein |
| LOF | 7 | klassisches ML (Dichte) | KI (weit) |
| Isolation Forest | 7 | klassisches ML (Ensemble) | KI (weit) |
| LSTM-Autoencoder | 7 | Deep Learning | KI (eng) |

MSTL, Z-Score und PELT optimieren keine Parameter aus Daten-Fehlern und
sind damit *keine* KI; LOF/IF lernen Strukturen datengetrieben (KI im
weiten Sinn); nur der LSTM-AE ist Deep Learning (KI im engen Sinn). Für
eine *Potenzialanalyse* ist diese Bandbreite gewollt — sie erlaubt den
Vergleich **entlang des KI-Grades**.

**Mustererkennung passiert auf zwei Ebenen.** Die in § 4.1 als „deskriptive
Mustererkennung" bezeichnete MSTL-Zerlegung legt die Saisonalität *explizit
und klassisch* offen (Varianten B/C); der LSTM-AE (§ 7.4) lernt das
Normalmuster *implizit*, um es zu rekonstruieren. In beiden Fällen gilt:
**Anomalie = Abweichung vom Muster** — die Anomalieerkennung setzt die
(klassisch oder gelernt) erfolgte Mustererkennung voraus. Die KI *betreibt*
somit Mustererkennung, nur ohne separaten Ausgabeschritt; dass MSTL den
expliziten Teil klassisch abdeckt, ist methodisch gewollt
(deseasonalisierter Eingang) und kein Widerspruch zum Titel. Die
begriffliche Einordnung erscheint ausführlich in der Gliederung § 2.0.

### 6.6 Begründung der Modellauswahl (Repräsentativität)

Die fünf Detektoren sind so gewählt, dass sie die **kanonischen
AD-Paradigmen** je einmal abdecken (Chandola u. a. 2009; Schmidl u. a. 2022) —
statistisch (Z-Score), dichtebasiert (LOF), isolations-/ensemblebasiert
(Isolation Forest), changepoint (PELT), rekonstruktions-/sequenzbasiert
(LSTM-AE). Für eine *Potenzialanalyse* ist **Repräsentativität wichtiger als
Architektur-Breite**: ein starker Vertreter je Familie hält den Vergleich
interpretierbar und ordnet jedes Verfahren einem Anomalietyp zu (H1; Lai
u. a. 2021).

**Warum nur ein Deep-Learning-Vertreter (LSTM-AE):**

1. Das **Datenregime** (univariat pro Zähler, niedrigdimensional, stark
   periodisch, ~35 k Stundenwerte, wenige Anomalien) ist genau das, in dem
   Klassik Deep Learning schlägt (Schmidl u. a. 2022; Kim u. a. 2022; Sehili
   und Zhang 2023) — mehr DL-Architekturen änderten das Fazit nicht.
2. Der **LSTM-AE ist der datennächste DL-Kanon** für saisonale, univariate
   Rekonstruktions-AD (Malhotra u. a. 2016 auf Basis Hochreiter und
   Schmidhuber 1997).
3. Die Alternativen sind entweder **Varianten derselben Rekonstruktionsidee**
   (VAE/Donut — Xu u. a. 2018) oder **multivariat-/korrelationsorientiert**
   (OmniAnomaly — Su u. a. 2019; GDN — Deng und Hooi 2021; Anomaly Transformer
   — Xu u. a. 2022). Letztere sind durch die nahezu unkorrelierten
   Zähler-Residuen (cross_meter_correlation, § 7.11/Ergebnisbericht) **nicht
   motiviert** → dokumentierter Ausblick, nicht umgesetzt.
4. **Fairness des KI-Vergleichs** kommt aus Tiefe statt Breite: erschöpfendes
   gridwh-HPO (§ 7.5), imbalance-robustes PR-AUC-Kriterium und Seed-Averaging
   am eingefrorenen Maßstab (§ 7.7) entkräften den Einwand „nur schlecht
   eingestellt" — ohne weitere Architekturen.

Damit ist „warum nicht mehr DL?" methodisch beantwortet: mehr DL wäre eine
*Benchmarking*-Arbeit; diese ist eine *Potenzialanalyse*. Ausführlich in
Gliederung § 2.2.4.

---

## 7. Stages 4–11 — Feature Engineering bis Robustheit

Stages 4–10 sind implementiert; ihre Beschreibung unten gibt den
**Ist-Stand** wieder. Stage 11 erfordert keinen neuen Code (erneuter
Pipeline-Lauf via `--dataset`).

### 7.1 Stage 4 — Feature Engineering *(implementiert)*

Implementiert in [src/stage4_features.py](../src/stage4_features.py).
Erzeugt zwei Feature-Tabellen auf dem stündlichen Stage-2-Index:

**Variante A** (`stage4_features_raw.parquet`) — Rohsignal + voller
Feature-Satz:
- **Pro Zähler:** `wmz_N_kw_mean` plus fünf rollierende Features
  (`rolling_mean_6h/24h`, `rolling_std_6h/24h`, `deviation_24h`) —
  rollierende Fenster-Statistiken sind etablierte Zeitreihen-Merkmale
  (Christ u. a. 2018)
- **Zyklische Kodierung:** `hour_sin/cos`, `weekday_sin/cos`,
  `month_sin/cos` — vermeidet 23↔0-Distanz-Artefakte (etablierte
  Feature-Engineering-Praxis ohne einzelne kanonische Quelle; in der
  Energieprognose verbreitet, z. B. zyklische Kodierung von
  Stunde/Wochentag/Monat plus Wetter-Features in Demir und Gunal 2025).
  Die sechs Formeln (Z. 102–107), seit 2026-08-14 auch in **MA 3.1.5**
  als Formelobjekte:

  ```
  hour_sin(t)    = sin(2π·h(t)/24)          hour_cos(t)    = cos(2π·h(t)/24)
  weekday_sin(t) = sin(2π·wd(t)/7)          weekday_cos(t) = cos(2π·wd(t)/7)
  month_sin(t)   = sin(2π·(mo(t)−1)/12)     month_cos(t)   = cos(2π·(mo(t)−1)/12)
  ```

  mit h(t) ∈ {0,…,23}, wd(t) ∈ {0,…,6} (Montag = 0), mo(t) ∈ {1,…,12}.
  **Das `− 1` beim Monat ist zwingend und darf beim Refactoring nicht
  verschwinden:** `index.hour` und `index.weekday` sind nullbasiert,
  `index.month` läuft aber von 1 bis 12. Ohne die Verschiebung begänne die
  Abbildung nicht bei 0, und Dezember und Januar lägen nicht benachbart auf
  dem Einheitskreis — also genau das 23↔0-Artefakt, das die Kodierung
  beseitigen soll, eine Zeitebene höher. In der MA sind die Symbole `wd`
  und `mo` statt `d` und `m` gewählt, weil `d_24(t)` dort schon die
  Abweichung vom Tagesniveau und `w` die Fensterlänge bezeichnet.
- **Binäre Flags:** `is_weekday`, `is_weekend`, `is_holiday`
  (deutsche Feiertage Berlin via `holidays.DE(subdiv="BE")`),
  `is_night` (22–05 Uhr)
- **Wetter:** `temperature`, `humidity`

**Variante B** (`stage4_features_residual.parquet`) — MSTL-Residuum +
minimaler Kontext:
- **Pro Zähler:** `wmz_N_residual` + `wmz_N_trend` (Skalen-Kontext) +
  `wmz_N_residual_rolling_std_24h` (Plateau-Detektor-Feature, da
  Varianz-Verlust ein Plateau signalisiert)
- **Geteilt:** `is_holiday` (irregulär, nicht von MSTL erfasst) +
  `temperature`. Cyclic-Encodings und Wochentag-Flags werden bewusst
  **nicht** mitgegeben, da MSTL die Saisonalitäten bereits aus dem
  Signal entfernt hat.

Beide Tabellen führen zusätzlich die Meta-Spalten `wmz_N_was_flagged`
und `wmz_N_interpolated` mit — **kein Modell-Input**, sondern für die
Trainings-Daten-Filterung in Stage 7.

**Feature-Zahl für Variante A: 17 zusätzlich zur Momentanleistung**
(nachgezählt 2026-08-14, damit die alte 16↔17-Frage erledigt ist):
10 Kalender (6 zyklische + `is_holiday` + `is_weekday` + `is_weekend` +
`is_night`) + 5 rollierende je Zähler + 2 Wetter. MA 3.1.5 nennt dieselbe
Zahl; wer hier Features ergänzt, muss sie dort mitziehen.

**Bewusst nicht enthalten: explizite Lag-Features.** Verschobene
Signalkopien (`x_{t-1}`, `x_{t-24}`, `x_{t-168}`) sind ein üblicher
Baustein der Zeitreihen-Merkmalskonstruktion, wurden hier aber
verworfen. Die ergebnisneutrale Seiten-Ablation
[src/tools/lag_ablation.py](../src/tools/lag_ablation.py) (IForest
raw/wmz_1, identische Best-HPs, Schwelle = Val-q99, point-adjusted F1;
zuletzt reproduziert 2026-07-15 auf dem Mac, identische Werte)
begründet das empirisch: F1 0,911 → 0,923 (+0,012, reiner
Precision/Recall-Tausch: P 0,88→0,94, R 0,95→0,91), ROC-AUC +0,004,
PR-AUC +0,007; auf Ereignis-Ebene regredieren drop (8/8→6/8) und spike
(12/12→10/12), plateau bleibt 0/4. Die Rolling-Features decken den
zeitlichen Kontext also bereits ab. In der Arbeit dargestellt in
Abschnitt 3.1.5 (Feature Engineering).

### 7.2 Stage 5 — Train/Val/Test-Split *(implementiert)*

Implementiert in [src/stage5_split.py](../src/stage5_split.py).
Einmalige, zeitlich strikte Aufteilung (linksgeschlossen, rechtsoffen) —
gilt identisch für alle Varianten und Modelle. Bei Zeitreihen ist eine
chronologische statt zufällig gemischter Aufteilung erforderlich, um
Look-ahead-Leakage zu vermeiden (Bergmeir und Benítez 2012):

| Split | Zeitraum | Anteil |
|---|---|---|
| Train | 2019-11-19 → 2022-10-31 | ~80 % |
| Val | 2022-11-01 → 2023-04-30 | ~10 % |
| Test | 2023-05-01 → 2023-11-19 | ~10 % |

Materialisiert als `outputs/<dataset>/parquet/split_assignment.parquet`
mit Index `timestamp` und Spalte `split` (Categorical).

### 7.3 Stage 6 — Normalisierung *(implementiert)*

Implementiert in [src/stage6_normalize.py](../src/stage6_normalize.py).
`StandardScaler`-Logik (Mean/Population-Std, `ddof=0` für Kompatibilität
mit scikit-learn, Pedregosa u. a. 2011) **pro Variante separat**, fit
ausschließlich auf Train-Zeilen, transform auf Train/Val/Test. NaN bleibt NaN (kein
Platzhalter); konstante Spalten erhalten `std=1`. Scaler werden nach
`outputs/<ds>/scalers/scaler_<variant>.parquet` persistiert. Erzeugt
`stage6_normalized_{raw,residual,trend}.parquet` (inkl. durchgereichter
Meta- und `split`-Spalten). Variante C (`trend`) skaliert ausschließlich
die drei `wmz_N_trend`-Spalten aus Stage 3.

### 7.4 Stage 7 — Modelltraining *(implementiert)*

Orchestrierung in [src/stage7_train.py](../src/stage7_train.py),
Detektoren in [src/models/](../src/models/). Es werden alle
`(Variante × WMZ × Modell)`-Jobs aus `models.registry` iteriert:

| Variante | Modelle | Detektor-Dateien |
|---|---|---|
| raw (A) | Z-Score, LOF, Isolation Forest, LSTM-AE, **Constancy** | `zscore/lof/iforest/lstm_ae/constancy.py` |
| residual (B) | Z-Score, LOF, Isolation Forest, LSTM-AE | dieselben |
| trend (C) | PELT, LSTM-AE | `pelt/lstm_ae.py` |

Quellen der Detektoren: Z-Score als statistische Baseline (Chandola u. a.
2009), LOF (Breunig u. a. 2000), Isolation Forest (Liu u. a. 2008),
PELT (Killick u. a. 2012); der LSTM-Autoencoder kombiniert LSTM
(Hochreiter und Schmidhuber 1997) mit dem rekonstruktionsfehler-basierten
Anomaliedetektions-Schema von Malhotra u. a. (2016).

**Constancy-Detektor (typ-spezifische Ergänzung, nur Variante A).** Die fünf
o. g. Paradigmen sind punkt- bzw. rekonstruktionsbasiert und übersehen
**Plateaus** systematisch (kollektive Anomalie: jeder Wert plausibel, nur die
Konstanz anomal → Plateau-Recall = 0 bei allen vieren, § 4.4.2). **Als
direkte Reaktion auf diesen empirischen Befund** wurde der
`ConstancyDetector` ([src/models/constancy.py](../src/models/constancy.py))
entwickelt; er schließt diese Lücke score-basiert: `score = flatness × collapse`, wobei
`flatness = exp(−local_std/std_scale)` die momentane Konstanz und
`collapse = 1 − exp(−max(base_std − local_std, 0)/(sensitivity·std_scale))`
den **Varianz-Einbruch gegenüber der jüngsten Eigen-Historie** des Zählers
misst (`base_std` über ein langes, um `window` zurückversetztes Kontext-
fenster). Dieser baseline-relative Aktiv-Kontext operationalisiert „konstant,
*obwohl* aktiv erwartet" ohne absolutes Level-Gate (nötig wegen der WMZ-
Heterogenität: wmz_2 Baseload ohne Off-Zustand, wmz_3 große Sommer-Null);
legitime Off-Phasen waren bereits flach → kein Einbruch → kein Flag.
Verfahrensfamilie: Varianz-Change-Point (Inclán und Tiao 1994; Killick u. a. 2012), Matrix Profile
(Yeh u. a. 2016), SAX (Lin u. a. 2007). Rein statistisch, deterministisch,
**CPU-only** (kein GPU) — damit on-the-edge-tauglich (Gliederung § 5.5.3).
HPO (`window`, `baseline_window`, `sensitivity`) wird **plateau-spezifisch**
selektiert (ROC-AUC gegen die Plateau-only-Ground-Truth, analog zur PELT-
F1-Sonderbehandlung); die Stage-10-Bewertung bleibt am eingefrorenen
Gesamt-Lineal (§ 7.7). Ergebnis: Plateau-Recall **1,0 / 1,0 / 0,0** (wmz_1 /
wmz_2 / wmz_3) gegenüber **0** bei allen Bestandsdetektoren; wmz_3 = 0 ist
korrekt, da dessen Test-Plateaus in der saisonalen Null liegen (~0 kW, nicht
von der legitimen Heizungs-Abschaltung trennbar). Gerahmt unter **H1**
(Methoden-Matching), nicht als neue Hypothese.

**Herleitung der Score-Funktion.** Die Score-Form folgt aus der Zerlegung der
Detektionsbedingung „konstant, *obwohl* aktiv erwartet" in zwei **unabhängige
Teilbedingungen**, die je auf einen stetigen Kern in [0, 1] abgebildet werden:

1. *Ist das Signal momentan flach?* → `flatness = exp(−local_std/std_scale)`.
   Gefordert ist eine monotone Abbildung „kleine lokale Streuung → 1, typische
   → 0". Der Exponentialkern `exp(−x)` ist die kanonische Wahl: beschränkt auf
   [0, 1], Wert 1 im Ursprung, überall differenzierbar, **ein** Skalenparameter
   (dasselbe Kern-Motiv wie bei RBF-/Gauß-Kernen — „wie viele typische σ an
   Variabilität liegen vor"). Die Normierung durch `std_scale` (robuster Median
   der positiven lokalen Std) macht den Term **skalenfrei** und damit über die
   heterogenen Zähler hinweg vergleichbar (wmz_1 ~10 kW vs. wmz_3 ~200 kW).
2. *Ist diese Flachheit ein Einbruch gegenüber der eigenen Historie?* →
   `collapse = 1 − exp(−(base_std − local_std)⁺/(sensitivity·std_scale))`. Hier
   ist die **komplementäre** Sättigungsrampe `1 − exp(−x)` (0 bei keinem Abfall,
   → 1 bei großem Abfall) gefordert; `()⁺` (Clip ≥ 0) lässt nur Varianz-*Ein-
   brüche* zählen, keine Anstiege.

Die **multiplikative** Verknüpfung `flatness × collapse` realisiert ein
**weiches logisches UND** (Produkt-T-Norm der Fuzzy-Logik):
nur wenn *beide* Terme hoch sind, ist der Score hoch — ist einer ≈ 0, ist der
Score ≈ 0, unabhängig vom anderen. Das ist der diskriminierende Mechanismus:
bei der legitimen saisonalen Abschaltung ist `flatness ≈ 1`, aber `collapse ≈ 0`
(kein Einbruch — der Zähler war *bereits* flach) → Score ≈ 0, kein False
Positive. Eine additive/mittelnde Verknüpfung wäre **kompensatorisch** (OR-
artig) und würde eine legitim-konstante Phase fälschlich auf ~0,5 heben.

Die Form ist eine **prinzipiengeleitete Heuristik**, iterativ gegen konkrete
Fehlschläge entwickelt: ein **absolutes Level-Gate** (v1) scheiterte an der
WMZ-Heterogenität (ein gehaltener Aktivwert fiel knapp unter die Niveau-
Schwelle), ein **niveau-basiertes Sigmoid-Gate** (v2) verfehlte Plateaus bei
niedrigem aktivem Wert; erst der **baseline-relative Varianz-Vergleich** (v3,
oben) löst beides, weil er nicht das *Niveau*, sondern die *Varianz gegen die
Eigen-Historie* prüft — die lokal-gleitende Form der Varianz-Change-Point-Idee
(Inclán und Tiao 1994; Killick u. a. 2012). Die Hyperparameter (`window`,
`baseline_window`, `sensitivity`) werden nicht gesetzt, sondern per HPO
(Stage 8) selektiert.

Gemeinsames API ([src/models/base.py](../src/models/base.py)):
`fit` / `score` (höher = anomaler; NaN-Zeile → NaN-Score) / `save` /
`load`. Wichtige Implementierungsdetails:

- **Per-WMZ-Slicing** (`model_features`): ein Modell für `wmz_2` sieht
  nur dessen Signal- und Rolling-Spalten plus die geteilten
  Time-/Wetter-Features — niemals die Spalten anderer Zähler
  (kein Cross-Leakage). Variante C ist univariat (nur `wmz_N_trend`).
- **Trainings-Daten-Filterung** (Methodik 6.4): Stunden mit
  `wmz_N_was_flagged == True` werden vor `fit` entfernt.
- **PELT** liefert keine kontinuierlichen Scores, sondern eine binäre
  Change-Point-Serie (1 an Change-Points). `ruptures.Pelt` auf dem
  univariaten Trend.
- **LSTM-Autoencoder** (PyTorch): Sequenz-Autoencoder über gleitende
  Fenster (`window_size`, Default 24); Score einer Stunde =
  Rekonstruktions-MSE des auf ihr endenden Fensters. Fenster mit NaN
  und die ersten `window_size−1` Stunden erhalten NaN. Seit 2026-07-09
  ist die Fehler→Score-Abbildung schaltbar (`score_mode`: `window_mse`
  = Default und Stage-10-Stand, `channel_mse`, `last_step_mse`,
  `mahalanobis` nach EncDec-AD, Malhotra u. a. 2016) — ausschließlich
  für die Score-Ablation in § 7.12; alle Hauptergebnisse laufen
  unverändert mit `window_mse`. Die Fenster-
  Tensoren werden in **`float32`** (NN-Standard) auf das Device gelegt —
  das halbiert den Speicher gegenüber `float64` und vermeidet auf MPS den
  pro-Operation-CPU-Fallback für `float64`. **Device-Wahl automatisch**
  (`device="auto"` → CUDA vor MPS vor CPU), sodass dasselbe Modell auf der
  RTX-3080-Ti (CUDA) wie auf Apple-Silicon (Metal/MPS) trainiert.
  **Praxis-Hinweis:** Auf Apple-MPS ist der `nn.LSTM`-Kernel in PyTorch
  derzeit (≤ 2.12) defekt — er lässt den Speicher explodieren (OOM bei
  > 60 GB), unabhängig vom `float32`-Fix. Das LSTM-Training läuft daher
  auf der CUDA-GPU (RTX 3080 Ti); MPS ist nur für die klassischen Pfade
  bzw. kleine Smoke-Tests nutzbar. Das gewählte Device wird nach jedem `fit()`
  im Stage-7- und Stage-10-Log als `[cuda]` / `[mps]` / `[cpu]`-Tag
  hinter dem Modellnamen ausgegeben, sodass auf einen Blick
  überprüfbar ist, dass der LSTM-Trainings-Pfad tatsächlich GPU
  verwendet (und nicht stillschweigend auf CPU zurückfällt).
  Anmerkung zur Windows-Task-Manager-Anzeige: der Standard-„3D"-
  Reiter zeigt die LSTM-Last nicht — das Engine-Dropdown muss auf
  `Cuda` / `Compute_0` gestellt werden. CUDA-Auslastung lässt sich
  unabhängig per `nvidia-smi` verifizieren (siehe Erfolgskontrolle in
  HANDOFF.md §10).
- **Lazy Imports:** `torch`, `scikit-learn` und `ruptures` werden erst
  in den fit/score-Methoden importiert. Damit bleibt die Pipeline (und
  die klassischen Detektoren) auf Maschinen ohne diese Pakete bzw. ohne
  GPU lauffähig.
- **Persistenz:** `outputs/<ds>/models/<variant>/<wmz>/<name>.pkl`.

**Kleinere Datensätze / Probelaufe:** `--max-train-rows N` begrenzt das
Training auf die jüngsten N Trainings-Stunden (zusammenhängender
Tail-Ausschnitt, kein Zufalls-Subsampling — die LSTM-Fensterbildung
braucht aufeinanderfolgende Stunden). Nützlich für schnelle Iteration
oder schwächere Hardware. Weitere CLI-Filter: `--variants`, `--wmz`.

**Reproduzierbarkeit:** Z-Score/LOF/IF/PELT sind deterministisch
(fester `random_state` bei IF). Der LSTM-AE ist auf CPU exakt seedbar,
auf CUDA/MPS jedoch nur bis auf nichtdeterministische GPU-Kernel — er
ist damit der einzige nicht bit-identisch reproduzierbare Baustein.

### 7.5 Stage 8 — Hyperparameter-Optimierung *(implementiert)*

Implementiert in [src/stage8_hpo.py](../src/stage8_hpo.py). Da die Modelle
unüberwacht sind, fehlt auf realen Daten ein Zielsignal. Lösung: dieselbe
Injektions-Maschinerie wie Stage 9 wird auf die **Validation**-Menge
angewandt — mit einem **eigenen Seed** (`VAL_SEED`), der sich vom
Test-Seed unterscheidet, sodass die HPs nicht auf die exakten
Test-Anomalien getunt werden. Das Test-Set bleibt während HPO unberührt
(kein Leakage).

Grid-Search pro `(Variante × WMZ × Modell)`; Auswahlkriterium:

- **Score-basierte Modelle** (Z-Score, LOF, IF, LSTM-AE): threshold-freie
  **ROC-AUC** gegen die injizierte Ground-Truth.
- **PELT** (binäre Change-Points): **point-adjusted F1** gegen die
  nicht-stationären Events.

**Suchstrategie — hybrid (Bergstra und Bengio 2012):** Die niedrig-
dimensionalen, billigen klassischen Modelle werden per **erschöpfendem
Grid** durchsucht; der hochdimensionale, teure LSTM-AE per **Random
Search** (bei gleichem Budget effizienter, sobald nicht alle Dimensionen
gleich wichtig sind).

| Modell | Strategie | durchsuchter Raum | Kombis |
|---|---|---|---|
| Z-Score | Grid | `aggregation` ∈ {max, l2, mean} | 3 |
| LOF | Grid | `n_neighbors` ∈ {5, 10, 20, 40, 80} | 5 |
| IsolationForest | Grid | `n_estimators` ∈ {100, 200, 400} × `max_features` ∈ {0.5, 0.7, 1.0} | 9 |
| PELT | Grid | `penalty` ∈ {5, 10, 20, 30, 50, 75, 100} | 7 |
| LSTM-AE | Random Search | `window_size` ∈ {24, 48, 72} × `hidden_size` ∈ {16, 32, 64} × `n_layers` ∈ {1, 2} × `learning_rate` ∈ {1e-2, 3e-3, 1e-3, 3e-4} × `epochs` ∈ {30, 50, 100} | 25 von 216 |

Das Random-Search-Sampling ist mit festem Seed (`SEARCH_SEED`)
reproduzierbar, zieht eindeutige Konfigurationen (keine Duplikate) und
verwendet dieselbe Kandidaten-Menge für alle LSTM-Jobs (vergleichbar).
Die Anzahl Ziehungen ist über `--lstm-search-iter` steuerbar (Default 25).

**HPO-Option A — gezielter `window × hidden`-Grid** (`--lstm-strategy
gridwh`). Der Daten-Sweep mit `window=48` erreichte auf `trend/wmz_1`
F1 = 0,776, während das Random-Search-HPO `window=24` wählte (F1 = 0,611).
Um empirisch zu klären, ob `window_size` die unter-abgetastete Dimension
des Random Search war, lässt sich Stage 8 mit `--lstm-strategy gridwh`
auf einen **erschöpfenden 3 × 3-Grid** `window_size ∈ {24, 48, 72} ×
hidden_size ∈ {16, 32, 64}` umstellen; die übrigen HPs werden auf
empirisch gute Defaults (`n_layers=2`, `learning_rate=1e-3`, `epochs=50`)
fixiert. Das sind 9 Configs/Job (81 fits über alle 9 LSTM-Jobs) statt
25 Random-Ziehungen — auf Apple-Silicon (M4 Pro, MPS) bzw. der
RTX 3080 Ti je ein kurzer Lauf. Default bleibt `--lstm-strategy random`;
`gridwh` ist ein bewusst gewählter Diagnose-Lauf, dessen Ergebnis die
Diskussion zur HPO-Budget-Frage trägt.

**Auswahl-Kriterium und Seed-Varianz** (`--hpo-metric`, `--hpo-seeds`).
Der gridwh-Lauf zeigte (Ergebnisbericht § 15.7), dass nicht das Such-Budget,
sondern das **Selektionskriterium** der Engpass ist: Die threshold-freie
Validierungs-ROC-AUC trennt die Konfigurationen kaum (alle ~0,50–0,52),
weil sie bei seltenen Anomalien über beide Klassen mittelt. Stage 8 erlaubt
daher alternative Kriterien für die score-basierten Modelle:
`--hpo-metric {roc_auc, pr_auc, f1}`. **PR-AUC** (Average Precision)
fokussiert die seltene Positiv-Klasse und ist bei Ungleichgewicht
trennschärfer (Davis und Goadrich 2006; Saito und Rehmsmeier 2015); **F1**
(point-adjusted, an der Val-Quantil-Schwelle) richtet die HPO direkt auf
die Testmetrik aus — F1-Maximierung über die Schwelle nach Lipton u. a.
2014; dass ein ROC-AUC-Optimum F1 nicht zwangsläufig maximiert, zeigen
Davis und Goadrich 2006. Ergänzend mittelt
`--hpo-seeds N` jede Konfiguration über N `random_state`-Werte
(Seed-Averaging) und dämpft so die Init-/Sampling-Varianz, die bei
stochastischen Netzen die HP-Rangfolge kippen kann (Reimers und Gurevych
2017; Bouthillier u. a. 2021). Die Selektion in Stage 8 erfolgt über den
Seed-**Mittelwert**; ins `hpo_log.csv` wird seit dem Patch 2026-06-11
zusätzlich **eine Zeile pro Seed** geschrieben (Spalte `seed`), damit die
Inter-Seed-Streuung der Best-HP-Wahl in der Post-HPO-Diagnose (§ 7.12)
als echte Std/Min/Max aus dem Log aggregiert werden kann — vorher war die
Std trivial 0, weil nur der Mittelwert persistiert wurde. Default bleibt
`roc_auc` mit einem Seed (rückwärtskompatibel); PR-AUC/F1 + Seed-Averaging
sind der in § 15.7 empfohlene nächste Schritt.

*Warum zunächst ROC-AUC?* Die threshold-freie ROC-AUC war die sachlich
begründete Default-Wahl: Sie bewertet die **Ranking-Trennschärfe** der Scores
**unabhängig vom Schwellwert** (der separat als Quantil gesetzt wird, § 7.7),
ist der etablierte Standard und war ohnehin schon Bestandteil der Stage-10-
Metriken. Dass sie bei *sehr seltenen* Anomalien flach wird, ist eine
**empirische Entdeckung** (§ 15.7), kein a-priori offensichtlicher Fehler —
und deckt sich mit der allgemeinen Lehre, dass ROC-AUC bei starkem
Klassen-Ungleichgewicht trügerisch optimistisch ist (Saito und Rehmsmeier
2015). **PR-AUC behält die Threshold-Freiheit** (der eigentliche Grund für
ROC-AUC) **und repariert nur die Imbalance-Blindheit** — die Umstellung ist
also eine minimale, prinzipientreue Korrektur, kein Bruch. Über die hier
verwendeten point- und schwellwertbasierten Maße hinaus existieren
**range-/sequenzbewusste, threshold-unabhängige** Maße als weiterführender
Evaluations-Standard der Zeitreihen-AD — range-based Precision/Recall
(Tatbul u. a. 2018) und „Volume Under the Surface" (VUS, Paparrizos u. a.
2022); sie sind für eine vertiefte Evaluation bzw. den Ausblick relevant.

**Bewusst nicht getunt: `contamination`** (LOF/IsolationForest) — der
Parameter beeinflusst nur das interne `predict()`-Schwellwert-Attribut,
nicht das Ranking von `score_samples()`. Da die HPO rangbasierte ROC-AUC
nutzt und der finale Schwellwert in Stage 10 datengetrieben (Val-Quantil)
gesetzt wird, hat `contamination` keinen Effekt auf die Ergebnisse.

A/B werden gegen **stationäre**, C gegen **nicht-stationäre** Anomalien
bewertet (Schiene aus §6.3). Output:
`outputs/<ds>/hpo/best_hparams.json` + `hpo_log.csv`. CLI-Filter
`--variants/--wmz/--models` (z. B. `--models zscore lof iforest pelt`,
um den LSTM-AE auf Maschinen ohne GPU auszulassen). Beide Output-Dateien
werden über [src/result_io.py](../src/result_io.py) **nicht-destruktiv**
geschrieben (siehe § 7.7 / § 6.0.3 Punkt 5): ein Teilmenge-Lauf merged
seine Ergebnisse in das bestehende JSON/CSV, ohne die anderen Modelle zu
verlieren. `--fresh` erzwingt Komplett-Neuschrieb.

### 7.6 Stage 9 — Synthetische Anomalie-Injektion (Test-Set) *(implementiert)*

Gemeinsame Bibliothek [src/anomaly_injection.py](../src/anomaly_injection.py),
Anwendung in [src/stage9_inject.py](../src/stage9_inject.py) über den
geteilten Baustein [src/injection_apply.py](../src/injection_apply.py).
Synthetische Anomalie-Injektion mit bekannter Ground-Truth ist gängige
Evaluationspraxis in der Zeitreihen-Anomaliedetektion; die hier
verwendete Typ-Taxonomie orientiert sich an Lai u. a. (2021).
Injiziert wird mit eigenem **Test-Seed** (`TEST_SEED`, ≠ Val-Seed). Die
Störung wird auf Signal-Ebene eingebracht und die abgeleiteten
Rolling-Features **neu berechnet**, anschließend mit dem auf Train
gefitteten Scaler re-normalisiert (reines transform). Es werden die
**vollen** Reihen gespeichert (nur Test-Stunden gestört), damit der
LSTM-AE beim Scoren der ersten Test-Stunden sauberen Fenster-Vorkontext
hat.

Anomalietypen — disjunkt nach Stationarität (§6.3):

| Schiene | Typ | Wirkung | Ziel-Variante |
|---|---|---|---|
| stationär | Spike | kW × {3, 5, 10}, 1 h | A, B |
| stationär | Drop | kW × {0, 0.25}, 1–3 h | A, B |
| stationär | Plateau | konstanter Wert, 6–24 h | A, B |
| stationär | Leckage | +{15, 30} %, 48–168 h | A, B |
| nicht-stationär | Drift | +{0.5, 1} %/Tag, 14–30 d | C |
| nicht-stationär | Strukturbruch | +{10, 20} % permanent | C |

Für Variante B wird **dieselbe additive kW-Differenz** auf das Residuum
addiert (mit dem Betreuer abgestimmt, konsistent mit §6.3 — keine
MSTL-Re-Dekomposition nötig). Multiplikative Typen (Spike/Drop) werden
nur dort platziert, wo das Signal aktiv ist (≥ 30 %-Quantil), um das
Sommer-Null-Problem aus §5.4 zu vermeiden.

Ground-Truth (`stage9_ground_truth.parquet`), je WMZ:
`gt_stat_<wmz>` / `gt_stat_label_<wmz>` (stationär, Typ bzw.
Typ@Intensität), `gt_nonstat_<wmz>` / `gt_nonstat_label_<wmz>`
(nicht-stationär), `gt_known_sensor_issue_<wmz>` (Kategorie A aus
Stage-2-Flags), `gt_no_data_<wmz>` (kein valides Signal); global
`gt_known_regulatory` (Kategorie C, ±14 Tage um EnSikuMaV 2022-09-01;
im Test-Bereich leer, relevant für die Variante-C-Auswertung über die
volle Reihe).

### 7.7 Stage 10 — Evaluation *(implementiert)*

Implementiert in [src/stage10_evaluate.py](../src/stage10_evaluate.py),
Metriken in [src/evaluation.py](../src/evaluation.py). Trainiert jedes
Modell mit den besten HPs aus Stage 8 (sonst Defaults) auf den sauberen
Train-Zeilen, scort das injizierte Test-Set und wertet stratifiziert aus.
**Schwellwert:** q-Quantil (Default 0.99) der Scores auf der **sauberen**
Validierung — unabhängig vom Test-Set; PELT ist binär (Schwelle 0.5).

**Scores statt Labels — und warum.** Die Detektoren geben **keine Labels
aus, sondern kontinuierliche Anomalie-Scores** (`score()`, § 7.4: höher =
anomaler — Rekonstruktions-MSE beim LSTM-AE, Isolations-/Dichte-Score bei
IF/LOF, |z| beim Z-Score). Ein **Label** (`y_pred`, binär) entsteht erst in
der Evaluation durch Anlegen des datengetriebenen **Schwellwerts** an die
Scores. Diese Trennung ist eine direkte Konsequenz des **unüberwachten**
Settings und methodisch gewollt:
1. **Unüberwacht:** Im Realbetrieb gibt es kein Label-Signal; Anomaliedetektion
   ist im Kern ein **Ranking-/Quantifizierungs-Problem** („wie ungewöhnlich?"),
   keine direkte Klassifikation — die Modelle *können* nur einen Grad ausgeben.
2. **Threshold-Freiheit in der Bewertung:** Scores erlauben die
   schwellenfreien Maße **ROC-AUC/PR-AUC**, die die *Trennschärfe* unabhängig
   von einem willkürlichen Cutoff messen (Davis und Goadrich 2006; Saito und
   Rehmsmeier 2015). Ein fest verdrahtetes Label würde das verschenken.
3. **Wählbarer Arbeitspunkt:** Der Schwellwert lässt sich pro Einsatz auf den
   gewünschten Precision/Recall-Tradeoff stellen; der Score entkoppelt „wie
   anomal" von „Alarm ja/nein".
4. **Ranking für die qualitative Inspektion:** Scores liefern die Top-N-Listen
   (10b) — höchstbewertete Treffer zuerst.

Daraus folgt die **doppelte Metrik-Logik** unten: *label-basiert* (P/R/F1 nach
Schwellwert) **und** *score-basiert/threshold-frei* (ROC-AUC, PR-AUC). Die
**Ground-Truth-Labels** (Stage 9) dienen ausschließlich der Bewertung, nie als
Modell-Input. **Sonderfall PELT:** liefert *direkt* ein binäres
Change-Point-Label (kein kontinuierlicher Score) — daher ROC-/PR-AUC
degeneriert, Beurteilung nur über F1.

**Objektivität — die Bewertung ist eingefroren.** Zu unterscheiden sind das
*Selektionskriterium* (Stage 8, wie ein Modell/HP ausgewählt wird) und die
*Bewertungsmethode* (Stage 10, wie es beurteilt wird). Über **alle**
Experimente hinweg — klassisch vs. LSTM, lokal vs. global (§ 7.11), jedes
HPO-Kriterium (§ 7.5) und jede Daten-Sweep-Stufe (§ 7.8) — bleibt der
Bewertungsblock **identisch**: dasselbe **Test-Set** mit derselben
**deterministischen Injektion** (Stage 9, `TEST_SEED=99`), dieselbe
**Ground Truth** und dieselben **Metriken**. Variiert werden nur die
*Kandidaten* und ihr *Trainings-Input*, nie das *Lineal*. So bleibt jeder
Vergleich fair. Zwei Konsequenzen: (1) Der Point-Adjust überschätzt zwar
absolut (Kim u. a. 2022; Sehili und Zhang 2023), aber **alle Modelle
gleich** — das *Ranking* bleibt unverzerrt; ergänzend werden die
threshold-freien **ROC-AUC und PR-AUC** berichtet, damit kein einzelnes Maß
die Schlussfolgerung trägt. (2) Für ein *sauberes* A/B müssen auch die
**Trainingsbedingungen** konstant sein (insb. Hardware) — Vergleiche,
die das verletzen (z. B. § 15.7: Laptop-CUDA vs. Mac-CPU; die numerische
Präzision ist dagegen durchgängig `float32`, s. § 7.4), sind als Diagnose
und nicht als A/B gekennzeichnet.

**10a — Quantitativ:** Precision/Recall/F1 (**point-adjusted**,
Xu u. a. 2018, mit Diskussion der Überschätzung nach Kim u. a. 2022)
sowie die threshold-freien **ROC-AUC und PR-AUC** pro
`Modell × Variante × WMZ`, zusätzlich Recall je Anomalietyp und je
Typ@Intensität. ROC-/PR-AUC sind nur für die score-basierten Modelle
aussagekräftig; für das binäre PELT degeneriert (Beurteilung über F1). Bewertet wird auf den **eligiblen**
Test-Stunden (ohne `gt_no_data` und ohne Kategorie-A-Stunden), damit der
Kategorie-B-Wert nicht durch Sensor-Fehler verzerrt wird. Output:
`outputs/<ds>/reports/stage10_metrics.csv`.

**10b — Qualitativ:** Pro Modell ein CSV mit den Top-Detektionen
(`stage10_qualitative_<variant>_<wmz>_<model>.csv`) inkl. Ground-Truth-
Spalten für die manuelle TP/FP/Borderline-Bewertung; das Protokoll wird
**vor der Analyse** festgehalten. Ergänzend erzeugt
[src/tools/plot_qualitative_case.py](../src/tools/plot_qualitative_case.py)
das *visuelle* Pendant: einen Overlay-Plot je (Variante × WMZ × Detektor ×
Event) mit drei Ebenen — physikalisches Signal in kW (raw/residual: rohes
kW-Signal, trend: MSTL-Trend, beide per Scaler zurückgerechnet), injiziertes
Ground-Truth-Band und Detektor-Score samt Schwellwert. Schwelle und Scores
sind Stage-10-identisch (gleiche HPs, q-Quantil der sauberen
Validierungs-Scores, PELT binär bei 0.5), sodass die Fälle exakt zu den
berichteten Metriken passen. So wird ablesbar, ob der Alarm zeitlich/formal
zum Event passt (TP), das Band ohne Alarm bleibt (FN) oder außerhalb gefeuert
wird (FP). `--list` listet alle Events einer (Variante,WMZ) inkl.
Detektor-Urteil für die Fallauswahl.
[src/tools/qualitative_protocol.py](../src/tools/qualitative_protocol.py)
aggregiert das Ganze zum systematischen Fall-Protokoll
[docs/qualitative_evaluierung.md](qualitative_evaluierung.md): aggregierte
Recall-Matrix (Typ × Detektor, aus `stage10_metrics.csv`), Event-Detektions-
Matrix je WMZ (✓/✗ + Spitzen-Score ÷ Schwelle), False-Positive-Kandidaten je
Detektor und eine kuratierte Fallauswahl mit leeren Spalten Beobachtung/
Erklärung — die fachliche Interpretation bleibt bewusst manuell.

**Nicht-destruktives Schreiben.** Wie Stage 8 nutzt auch Stage 10 die
Helfer in [src/result_io.py](../src/result_io.py): `stage10_metrics.csv`
wird über `write_rows_csv()` als **Merge in Bestehendes** geschrieben
(Schlüssel `variant × wmz × model`, kollidierende Zeilen werden ersetzt,
nicht-betroffene bleiben erhalten). Ein Teilmenge-Lauf wie
`stage10_evaluate.py --models lstm_ae` ersetzt damit nur die 9
LSTM-AE-Zeilen pro Stratum und behält die Klassik-Einträge unverändert
— Voraussetzung für die in § 6.0.3 (Punkt 5) beschriebene iterative
Experiment-Choreografie. `--fresh` erzwingt Komplett-Neuschrieb.

**Empirischer Stand der Auswertung (2026-06-02):** Die Stages 7–10 sind
für alle fünf Detektoren vollständig durchgelaufen (Klassik:
Grid-Search; LSTM-AE: 25-stufiger Random-Search über das Gitter aus
§ 7.5, ~3 h 30 min Wandzeit auf einer RTX 3080 Ti Mobile). Die
Headline-Ergebnisse sind im
[Ergebnisbericht](ergebnisbericht.md)
§ 11 / § 15 dokumentiert. Knapp: **IsolationForest dominiert die raw-
Schiene** (`raw/wmz_1` F1 = 0,911), **LOF dominiert die residual-
Schiene** (`residual/wmz_1` F1 = 0,837), **PELT führt auf der
trend-Schiene** (`trend/wmz_1` F1 = 0,700). Der LSTM-Autoencoder
kommt in keinem der neun (Variante × WMZ)-Setups an den jeweiligen
Klassik-Champion heran; am nächstgelegenen ist `trend/wmz_1`
(F1 = 0,611). **Auf Typebene** schneidet der LSTM-AE für
Drift (5/8 vs. PELT 4/8) und Strukturbruch (2/3 vs. 1/3) sogar besser
ab als PELT — die Gesamt-F1 fällt nur durch PELTs deutlich höhere
Precision (0,997 vs. 0,712) zurück. Methodisch diskussionswürdig
(siehe Ergebnisbericht § 15.6): das Random-Search-Budget hat im
Gitter über `window_size` nur 25 von 216 Konfigurationen abgetastet,
und der Daten-Sweep mit `window=48` (vom HPO nicht gewählt) liefert
auf `trend/wmz_1` schon bei 10 000 Trainings-Stunden F1 = 0,776 —
ein Hinweis, dass der wahre LSTM-AE-Wert mit grösserem HPO-Budget
oder einem Bandit-Verfahren (Hyperband/BOHB) deutlich höher liegen
könnte.

**10c — Labelfreie Realwelt-Validierung:** PELT wird auf dem vollen Trend
ausgeführt; `stage10_changepoints_regulatory.csv` listet die
Change-Points im ±30-Tage-Fenster um die EnSikuMaV (2022-09-01).
*Empirischer Befund im sauberen Datensatz:* für alle drei Zähler wurden
Change-Points nahe der EnSikuMaV gefunden — starkes Argument für die
Trend-Schiene.

### 7.8 Datenmengen-Sensitivität — Daten-Sweep *(Diagnose-Werkzeug)*

Ergänzendes Diagnose-Werkzeug zu den eigentlichen Pipeline-Stages:
[src/tools/data_sweep.py](../src/tools/data_sweep.py) trainiert einen einzelnen
Detektor mit **identischen Hyperparametern** auf unterschiedlich großen
Subsets des Trainings-Tails und bewertet jeweils auf demselben
(injizierten) Test-Set. Die Stufen werden in einer kumulativen
CSV-Datei (`outputs/<ds>/reports/data_sweep.csv`) gesammelt, sodass
Sweeps für verschiedene `(Variante × WMZ × Modell)`-Kombinationen
nebeneinander stehen.

**Methodische Grundlage.** Stage 5 fixiert Validierung und Test
zeitlich (kein Shuffling, kein Sampling). Variiert man nur die
Trainingsmenge, bleiben Val und Test bit-identisch über alle
Sweep-Stufen — die Performance-Differenzen sind dadurch sauber der
Trainingsdatenmenge zuzuordnen. Das ist die klassische
Learning-Curve-Diagnose (Cortes u. a. 1994; Perlich u. a. 2003;
Viering und Loog 2023), angewandt auf unüberwachte Anomalie-Detektoren.
Der Sweep variiert damit **genau eine Achse (Datenmenge)** am
eingefrorenen Bewertungs-Maßstab (§ 7.7) — ein kontrolliertes
Ein-Variablen-Experiment, komplementär zur Kriteriums-Achse (§ 7.5,
Option E) und zur Lokal-vs-Global-Achse (§ 7.11, Option F).

**Verzahnung mit dem globalen Training (Option F) und dem HPO-Kriterium
(Option E).** Der Sweep beantwortet „**ist das LSTM-AE datenlimitiert** —
steigt die Lernkurve noch oder ist sie gesättigt?" und ist damit der
**Vorab-Test für das Pooling (§ 7.11):** nur wenn die Kurve noch steigt,
verspricht *mehr* Daten überhaupt Gewinn. Wichtig ist die Art der Daten:
der Sweep fügt mehr Daten **desselben** Zählers hinzu (längere Historie,
triviale Gleichverteilung) und misst so den Grenznutzen einer
gleichverteilten Stichprobe; Pooling fügt Daten **anderer** Zähler hinzu
und *wettet*, dass diese nah genug an derselben Verteilung liegen. Pooling
lohnt daher nur, **wenn beide Diagnosen zutreffen**: Kurve steigt noch
(Sweep) **und** die Zähler sind strukturell ähnlich/nicht redundant
(Korrelationsanalyse, § 7.11 / `cross_meter_correlation.py`). Außerdem
liefert der Sweep den **diagnostischen Zielwert für Option E**: er hat
gezeigt, dass `window=48` auf `trend/wmz_1` mit viel Daten bis F1 = 0,776
erreicht (§ 15.6/§ 15.7), das vom ROC-AUC-Kriterium aber nicht gewählt wird.
Da der Sweep dafür die **Test-F1** nutzt, ist er ein **exploratives
Diagnose-Werkzeug, keine Selektionsmethode** (sonst Test-Peeking) — die
faire Frage „kann ein *prinzipielles* Kriterium diese Konfiguration
*ohne* Test-Blick treffen?" beantwortet erst Option E.

**Trainings-Subset-Konvention.** Genommen wird der **jüngste
zusammenhängende Tail** der Flag-bereinigten Trainings-Indizes, kein
Zufalls-Subsampling. Begründung:

1. Der LSTM-AE bildet gleitende Fenster aus aufeinanderfolgenden
   Stunden — Zufallsstichproben würden die Sequenz-Struktur
   zerstören und das Modell unbrauchbar machen.
2. Der Subset liegt zeitlich am nächsten an Val/Test, was die
   praxisrelevante Frage „wie wenige *aktuelle* Daten reichen?"
   beantwortet (im Gegensatz zu „irgendein zufälliges historisches
   Stück").
3. Klassiker und LSTM-AE trainieren auf exakt denselben Stunden,
   was den modellübergreifenden Vergleich fair hält.

**Hyperparameter-Quelle.** Der Sweep lädt per Default die HPs aus
`outputs/<ds>/hpo/best_hparams.json` für (variant, wmz, model) — das
sind dieselben Werte, mit denen Stage 10 evaluiert. Damit ist die
Sweep-Stufe `train_rows=0` (volle Trainingsmenge) direkt mit der
Stage-10-Zeile derselben Konfiguration vergleichbar. Mit
`--hp key=value` lassen sich einzelne HPs überschreiben (z. B. für
Ablationen), mit `--no-load-best` werden die Modell-Defaults
verwendet.

**Schwellwert-Konvention.** Wie in Stage 10: PELT ist binär
(Schwellwert 0,5), score-basierte Modelle nutzen das 99-%-Quantil der
sauberen Validation-Scores. Damit ist die Sweep-Metrik direkt mit
der Stage-10-Tabelle vergleichbar.

**Output-Format.** Eine Zeile pro (Sweep-Aufruf × Train-Stufe). Jede
Zeile enthält neben den Metriken (P/R/F1/ROC-AUC) auch die
tatsächlich verwendeten HPs als `hp_<key>`-Spalten, die effektive
Trainings-Stundenzahl, den Anteil an der vollen Trainingsmenge, die
fit/score-Dauer und das verwendete Device. Damit ist jede Zeile
für sich genommen reproduzierbar.

**Erkenntnisse aus den Champion-Sweeps (Stand 2026-06-02, alle
9 Klassik-Champions + 3 LSTM-AE-Champions abgedeckt).** Die kumulative
Sweep-CSV enthält 48 Stufen-Zeilen, aus denen vier nicht-triviale
modellabhängige Sättigungsmuster ableitbar sind (Details + Tabellen
im [Ergebnisbericht](ergebnisbericht.md)
§ 14.7.3 und § 14.7 LSTM-Block):

1. **PELT ist invariant gegenüber der Trainingsmenge.** Alle vier
   Stufen liefern identische F1/AUC. Konzeptionell konsistent: PELT
   wertet auf dem **vollen Trend** aus, nicht auf einem Train-Subset;
   die binäre Schwelle entkoppelt den Detektor zusätzlich von der
   Validation-Score-Verteilung.
2. **IsolationForest hat einen klaren Sättigungsknick** bei
   ~10 000 Trainings-Stunden für `raw/wmz_1` (F1 = 0,876), darunter
   ist die Baum-Vielfalt zu klein und der Detektor instabil; darüber
   gewinnt er nur noch wenige Prozentpunkte.
3. **LOF hat ein nicht-monotones Optimum unter der vollen Menge.**
   `residual/wmz_1` erreicht bei 5 000 Stunden F1 = 0,895 — höher als
   bei voller Trainingsmenge (0,837). Mehr Daten machen das
   Dichte-Modell „globaler" und schwächen die lokale Sensitivität.
4. **LSTM-AE zeigt starkes Overfitting bei voller Trainingsmenge.**
   Auf `raw/wmz_1` ist F1 mit 2 000 Trainings-Stunden = 0,765 (mit
   den vom HPO gewählten HPs), bei voller Menge fällt es auf 0,212;
   auf `trend/wmz_1` analog 0,772 → 0,611. Konsequenz: die HPO findet
   HPs, die bei kleinen Trainingsmengen exzellent sind, bei voller
   Menge aber overfitten. Ein gemeinsam getuntes (HPs × Datenmenge)-
   Optimum bleibt als methodische Erweiterung offen.

Die kumulative Sweep-CSV ist in der Thesis-Auswertung als ergänzendes
Diagnose-Artefakt einsetzbar (Datenmengen-Sensitivitäts-Plot pro
Modell), gehört aber **nicht** zur Haupt-Ergebnistabelle. Die
Haupt-Tabelle bleibt `stage10_metrics.csv` (Stage 10 mit den HPs aus
Stage 8 bei voller Trainingsmenge).

**Generalisierung auf wmz_2/wmz_3 (Klassik, Läufe 2026-06-01; in MA 4.3
seit 2026-07-17):** IForest raw bleibt auf beiden Zählern über alle
Stufen bei F1 ≈ 0 (0,000–0,007), LOF residual/wmz_2 schwankt
nicht-monoton (0,05 → 0,01 → 0,43 → 0,32), PELT ist stufenunabhängig
konstant (0,531 wmz_2; 0,000 wmz_3). Befund: Die schwachen Ergebnisse
auf wmz_2/3 sind **kein Datenmengen-Artefakt** — limitierend ist die
dokumentierte Datenqualität/Testbasis (MA 4.1). Die LSTM-AE-Seite der
Generalisierung (raw/trend × wmz_2/3) ist **durchgeführt 2026-07-20 auf
der RTX 3080 Ti** (CUDA, 4 Jobs × 4 Trainingsstufen, 6:11 min Wandzeit;
`python src/tools/data_sweep.py --model lstm_ae --variant {raw,trend}
--wmz {wmz_2,wmz_3}`): In allen vier Jobs bleibt
F1 an drei von vier Stufen bei exakt 0,000, nur trend/wmz_2 springt bei
voller Menge auf 0,211 (weiter unter PELT 0,531). Damit ist die
Datenmenge auch beim LSTM-AE kein Erklärer der Klassik-Überlegenheit;
die Lernkurve bricht auf wmz_2/3 bereits vor jeder HP-Interaktion
zusammen (Detail: ergebnisbericht § 13.2, in MA 4.3 seit 2026-07-20).

### 7.9 Stage 11 — Robustheits-Evaluation *(kein neuer Code nötig)*

Die komplette Pipeline ist über `--dataset` parametrisiert; Stage 11 ist
damit ein erneuter Durchlauf der Stages 1–10 auf einem „schmutzigen"
Datensatz, **ohne neuen Code**. Aktuell liegt **kein zweiter Datensatz**
vor; sobald er existiert, wird er als `data/<name>/` abgelegt und die
Pipeline mit `--dataset <name>` durchlaufen (siehe Sektion 8). Die
konkrete Definition des schmutzigen Datensatzes wird vorher festgelegt.

### 7.10 Ausblick — Anomaly Transformer

Anomaly Transformer (Xu u. a. 2022) als alternatives
Sequenz-Modell ist konzeptionell vorgesehen, aber für die
Hauptauswertung nicht eingeplant. Implementation als optionale
Erweiterung am Ende der Arbeit, falls Zeit verbleibt; sonst im
Ausblick-Kapitel als methodische Erweiterung erwähnt. Methodisch sinnvoll
wird er erst im **multivariaten** Setting (alle WMZ gleichzeitig als
Eingabe, s. § 7.11 Design b): sein Association-Discrepancy-Mechanismus
braucht korrelierte Kanäle, die er gegeneinander stellen kann — bei
univariater/lokaler Verarbeitung hat er nichts zu assoziieren.

**Multivariate AD adressiert eine andere Anomalieklasse.** Univariate
Detektion (pro Zähler) erfasst Punkt-, kontextuelle und kollektive
Anomalien (Chandola u. a. 2009; Lai u. a. 2021). Multivariate Verfahren —
Anomaly Transformer (Xu u. a. 2022), graphbasierte Modelle wie GDN (Deng
und Hooi 2021), stochastische RNN-VAEs wie OmniAnomaly (Su u. a. 2019) —
zielen zusätzlich auf **Inter-Signal-/Korrelationsstruktur-Anomalien**:
jedes Signal liegt einzeln im Normbereich, aber die *Beziehung* zwischen den
Signalen bricht. Diese Klasse existiert nur im gemeinsamen Raum und ist für
Per-Signal-Detektoren unsichtbar. Voraussetzung ist jedoch eine **stabile
Abhängigkeitsstruktur** zwischen den Kanälen — genau die fehlt hier: die
WMZ-Residuen sind nahezu unkorreliert (§ 7.11, `cross_meter_correlation.py`).
Dieser Anomalietyp ist im vorliegenden Ein-Gebäude-Datensatz daher praktisch
**nicht vorhanden**, weshalb multivariate Verfahren keinen Mehrwert
versprechen. Hinzu kommt, dass die Evaluation dieses Teilfelds in der
Kritik steht: das Point-Adjust-Protokoll überschätzt Deep-Modelle derart,
dass triviale Baselines bzw. Zufallsrauschen die SOTA schlagen (Kim u. a.
2022; Sehili und Zhang 2023). Relevant würde multivariate AD erst bei
gekoppelten, heterogenen Daten (mehrere Gebäude/Energieträger, § 7.9).

### 7.11 Ausblick — Globales vs. lokales Training (Cross-Meter-Sharing)

**Status quo (lokal):** Die Pipeline trainiert **pro WMZ getrennt** (§ 6.2,
`registry.iter_training_jobs()` erzeugt variant × wmz × modell;
`model_features()` schließt die jeweils anderen Zähler aus, um
Cross-Leakage zu vermeiden). Die raw/residual-Varianten sind dabei bereits
**multivariat** (per-WMZ-kW-Features + Zeit-Features + `temperature`/
`humidity` als shared features); nur `trend` ist univariat. „Multivariat"
(mehrere Kanäle inkl. exogener Wetterkovariate) ist jedoch zu unterscheiden
von „heterogen" im Quellen-Sinn (verschiedene Gebäude/Energieträger/
Qualitätsstufen) — der vorliegende Datensatz ist multivariat, aber von der
Quelle her homogen (1 Gebäude, Energieträger Wärme, 3 baugleiche WMZ).

**Ausblick-Variante (global; nicht umgesetzt, s. u.):** Ein **einziges** LSTM-AE über die Fenster
**aller drei WMZ** mit geteilten Gewichten; Scoring je Zähler. Dies ist das
einzige Setup, in dem ein KI-Modell einen **strukturellen Vorteil gegenüber
der Klassik** besitzt: Z-Score/LOF/IF/PELT haben keinen Mechanismus, über
Zähler hinweg zu lernen, ein globales Netz schon. Erwarteter Nutzen: (1) 3×
Trainingsdaten → stabilerer Fit, geringere Seed-Varianz; (2) geteilter
Prior → Transfer auf den schwächsten Zähler (wmz_1, F_Saison ≈ 0,30); (3)
implizite Regularisierung → weniger False Positives; (4) Skalierbarkeit
(1 Modell statt N). Theoretischer Rahmen: Montero-Manso und Hyndman 2021
(globale Modelle gewinnen v. a. bei *vielen* Serien; mit nur 3 Zählern
Effekt begrenzt, **negativer Transfer möglich**).

**Zwei Designs:** (a) *Pooled/global* — Fenster aller WMZ stapeln, gemein-
sames Feature-Schema oder Meter-ID-Embedding; saubere Ablation *lokal vs.
global* (Architektur fix, nur Daten-Sharing variiert). (b) *Multivariat-
gemeinsam* — alle WMZ als ein 3-fach-breiter Eingabevektor → detektiert
system-/gebäudeweite Anomalien und ist das Setting, in dem der Anomaly
Transformer (§ 7.10) passt.

**Empirischer Befund (Korrelationsanalyse, § 5.4
`cross_meter_correlation.py`).** Die paarweise Korrelation der drei WMZ
hängt stark von der Signalebene ab:

| Paar | Roh-kW (A) | MSTL-Trend | **Residuum (B)** |
|---|---:|---:|---:|
| wmz_1 ↔ wmz_2 | 0,41 | 0,61 | **0,06** |
| wmz_1 ↔ wmz_3 | 0,47 | 0,60 | **0,03** |
| wmz_2 ↔ wmz_3 | 0,74 | 0,83 | **0,35** |

Die Korrelation steckt fast ausschließlich im **Trend** (gemeinsamer
Saison-/Temperaturtreiber); das **anomalie-relevante Residuum ist nahezu
unkorreliert** (wmz_1 gegen beide ≈ 0). Zusätzlich liegen die Roh-Skalen
weit auseinander (wmz_3 mit Mittel ≈ 95 kW gegenüber wmz_1 ≈ 10 kW). Daraus
folgt für die Designs:

- **Pooling (a) ist nur auf der Residuum-Ebene (Variante B) gerechtfertigt:**
  dort ist die Redundanz niedrig (≈ 0 Korrelation → echtes Mehr-an-Daten)
  und die STL-Zerlegung + Normalisierung vereinheitlichen Skala und Saison
  → strukturell homogenere Eingaben. Auf der **Roh-Ebene (A)** ist Pooling
  ungünstig: sehr verschiedene Skalen/Formen (negativer Transfer, v. a. für
  den strukturell abweichenden wmz_1) **und** höhere Redundanz.
- **Multivariat-gemeinsam (b) / Anomaly Transformer ist von diesen Daten
  kaum motiviert:** auf der AD-relevanten Residuum-Ebene gibt es **keine
  zähler-übergreifende Korrelation auszunutzen**; die vorhandene Trend-
  Korrelation ist bloß gemeinsame Saisonalität, die ein Per-Zähler-Modell
  ohnehin abbildet.
- Die ≈-null Residuum-Korrelation ist zugleich ein **nachträglicher Beleg
  für die Per-Zähler-Architektur** (§ 6.2): die Abweichungen vom
  Normalmuster laufen je Zähler unabhängig.

**Pooling — konkrete Mechanik (Design a).** Pooling stapelt die Fenster
aller Zähler als **Zeilen** (mehr Beispiele), nicht als Spalten/Kanäle (das
wäre Design b). Das Modell verarbeitet weiterhin **ein Fenster nach dem
anderen**; die Zähler „treffen sich" nur im **gemittelten Gradienten eines
gemischten Mini-Batches** und damit in den **geteilten Gewichten** — es gibt
**keine Verbindung über die Datenachse** (kein Zeitabgleich zwischen
Zählern). Konzeptionell werden die Fenster als **austauschbare Stichproben
desselben Erzeugungsprozesses** behandelt. Daraus die Bausteine:

- **Per-Zähler-Normalisierung *vor* dem Pooling:** Jeder Zähler wird mit
  *seiner eigenen* Statistik standardisiert (sonst dominierte wmz_3 mit
  ≈ 95 kW das Modell allein durch die Größe). Erst dann sind die Fenster
  vergleichbar.
- **Optionales Meter-ID-Embedding:** ein gelernter „Welcher-Zähler"-Vektor
  ergibt einen **geteilten Rumpf + zählerspezifische Verschiebung** — der
  Mittelweg zwischen voll-lokal und voll-gepoolt und die Versicherung gegen
  negativen Transfer beim Ausreißer wmz_1.
- **Schwelle bleibt pro Zähler:** Auch bei global trainiertem Modell ist die
  Fehlerverteilung je Zähler verschieden → der Schwellwert wird weiterhin
  aus dem Quantil der sauberen Val-Scores **je Zähler** gesetzt. *Trainiert
  wird global, geschwellt wird lokal.*

**Detektions-Semantik und Ground Truth (Design a).** Die Klassifikation
bleibt **per Zähler** (ein Signal genügt): jedes injizierte Zähler-Fenster
wird einzeln durch das globale Modell gescort und gegen die **bestehende
per-Zähler-Ground-Truth** gemessen. Es ist **keine Neu-Injektion** nötig —
die Auswertung ist Stage-10-identisch und damit objektiv mit den lokalen
Modellen vergleichbar (§ 7.7). Nur Design (b) bräuchte **neue
Cross-Meter-Anomalietypen** (De-Korrelation/Inkonsistenz) und damit eine
geänderte Stage-9-Injektion — ein weiterer Grund, (b) im Ausblick zu halten.

**Selektionskriterium:** Das globale Experiment verwendet dasselbe
HPO-Kriterium wie § 7.5 (PR-AUC/F1 + Seed-Averaging) — sonst träfe es
denselben Kriteriums-Mismatch wie der lokale gridwh-Lauf (Ergebnisbericht
§ 15.7). **Stand (Entscheidung 2026-06-04): wird in dieser Arbeit nicht
umgesetzt** — die globale Variante bleibt **Ausblick** (Kap. 6), da der
Korrelationsbefund (oben) ihren Mehrwert auf das Residuum und auf einen
marginalen Effekt bei nur drei Zählern begrenzt. Die Spezifikation hier ist
fachliche Referenz für eine mögliche Folgearbeit; Design/Refactor-Aufwand in
HANDOFF.md § 9.2 (Option F).

### 7.12 Post-HPO-Diagnose-Werkzeuge

Fünf eigenständige Skripte unter [src/tools/](../src/tools/), die
nach dem Stage-7/8/10-Lauf auswerten, **ob** die HPO-Wahl belastbar war
und **warum** ein Modell die F1-Werte aus Stage 10 liefert, die es
liefert. Beides sind diagnostische Post-Pipeline-Tools, kein Bestandteil
der eigentlichen Detektions-Pipeline (§ 6.0.3 Punkt 6) — sie produzieren
keine neuen Modelle, sondern interpretieren vorhandene Outputs.

**Seed-Varianz** ([src/tools/seed_variance.py](../src/tools/seed_variance.py)).
Stage 8 erlaubt mit `--hpo-seeds N` Seed-Averaging je HP-Konfiguration
(siehe § 7.5) und schreibt seit dem Patch 2026-06-11 **eine Zeile pro
Seed** in `hpo_log.csv` (Spalte `seed`), nicht mehr nur den Mittelwert
— die Selektion in Stage 8 bleibt weiterhin der Mittelwert, geloggt wird
zusätzlich die Per-Seed-Granularität, damit die Inter-Seed-Streuung aus
dem Log rekonstruierbar ist (vorher trivial std = 0 bei einer Zeile pro
Config). Das Skript liest `hpo_log.csv`, gruppiert die LSTM-AE-Zeilen
nach `(variant, wmz, HP-Tupel)` und aggregiert über die `N` Seeds zu
**Mean ± Std**, Min, Max, sowie:

- `runner_up_mean` — Score der **zweitbesten** HP-Kombination im selben Job,
- `gap = best_mean − runner_up_mean` — Abstand der Best- von der
  Runner-up-HP,
- `robust_2sigma = (gap > 2·best_std)` — 2-Sigma-Heuristik, ob die
  Best-vs-Runner-up-Differenz größer als die Seed-Unsicherheit ist.

**Was die Auswertung sagt.** Ist `robust_2sigma` in der Mehrheit der
Jobs `False`, lebt die HP-Wahl statistisch im Init-Rauschen — die in
§ 7.5 zitierten Seed-Varianz-Befunde (Reimers und Gurevych 2017;
Bouthillier u. a. 2021) werden damit für die *konkrete Auswertung*
quantifiziert. Output: `outputs/<ds>/reports/lstm_seed_variance.csv`.
Aufgegriffen im Ergebnisbericht § 15.8 und in der Diskussion
zu L6 (§ 16: GPU-LSTM nicht bit-reproduzierbar).

**Score-Verteilungs-Diagnose** ([src/tools/score_distributions.py](../src/tools/score_distributions.py)).
Lädt für jede `(Variante × WMZ)`-Kombination das LSTM-AE-Modell und
den jeweiligen Klassik-Champion (IForest auf raw, LOF auf residual,
PELT auf trend), scort beide auf dem injizierten Test-Set
(`stage9_injected_<variant>.parquet`) und plottet pro Job die
Score-Verteilung getrennt nach Ground-Truth-Klasse (normal vs. anomal).
Per-Modell-Normalisierung (Min-Max nach Perzentil-Clipping) macht die
Verteilungs-Formen über Modell-Familien hinweg vergleichbar; die
Stage-10-Schwellen werden als gestrichelte Linien eingezeichnet.

**Wozu der Plot dient.** Die Stage-10-Tabelle (§ 7.7) sagt zwar *was*
ein Modell liefert (F1, AUC), aber nicht *warum*. Der Plot diagnostiziert
das dominante Muster pro Job als eine von vier Kategorien:

- **`inversion`** — Anomalien rangieren *unter* normalen Stunden
  (Mean(anomal) < Mean(normal)). Erklärt direkt ROC-AUC < 0,5
  (z. B. LSTM-AE auf trend laut Ergebnisbericht § 11.2). Strukturelles
  Modellversagen, nicht durch Schwellwert reparierbar.
- **`overlap`** — Verteilungen überlappen stark (|Mean-Abstand| < 0,05).
  Das Modell trennt schlecht; F1 bleibt unabhängig vom Schwellwert
  niedrig. Strukturelles Modellversagen, ebenfalls nicht reparierbar.
- **`threshold-shift`** — Verteilungen sind getrennt, aber das
  99-%-Val-Quantil liegt im falschen Bereich (außerhalb des
  Trennintervalls). **Fixbarer Defekt** — andere Schwelle würde
  helfen, das Modell selbst ist okay.
- **`ok`** — Verteilungen getrennt + Schwelle dazwischen. Erwartetes
  Verhalten eines funktionierenden Detektors.

Die Klassifizierung pro Job liegt in
`outputs/<ds>/reports/score_distribution_diagnose.csv`; das Bild in
`outputs/<ds>/figures/score_verteilungen.png`. Aufgegriffen im
Ergebnisbericht § 15.7 / § 15.8 als Erklärung der Stage-10-F1-Werte
(„welche der drei Schwächen liegt vor — Inversion, Overlap, oder nur
falsche Schwelle?").

**LSTM-AE-Rekonstruktions-Sicht** ([src/tools/lstm_ae_reconstruction.py](../src/tools/lstm_ae_reconstruction.py), ergänzt 2026-07-09).
Macht das *implizite* Normalmodell des LSTM-AE einsehbar und stellt es
dem *expliziten* Normalmodell der MSTL gegenüber: Für einen wählbaren
Zeitausschnitt werden Original-kW, AE-Rekonstruktion (letzter
Zeitschritt des auf der Stunde endenden Fensters — dieselbe Zuordnung
wie beim Score) und der MSTL-Fit (Trend + Saison 24 h + 168 h, nur
Variante raw) überlagert, dazu das mittlere Tagesprofil des
Ausschnitts; optional (`--latent`) eine PCA der Encoder-Latent-Vektoren,
eingefärbt nach Stunde und Wochentag. **Diagnostischer Zweck:** trennt
die zwei Erklärungen der LSTM-AE-Unterlegenheit — rekonstruiert der AE
den Tages-/Wochengang sauber (Latent-Raum nach Tageszeit geordnet) und
versagt *trotzdem* in der Anomalie-Trennung, liegt das Problem im
Scoring/der Schwelle (Kriteriums-Mismatch, § 7.5/§ 7.12
Score-Verteilungs-Diagnose); lernt er schon das Muster nicht, wäre der
Befund fundamentaler. Erster Lauf (raw/wmz_2, GPU-Checkpoint):
r(AE-Rekonstruktion, Original) ≈ 0,97, Latent-Raum klar nach
Tageszeit/Wochentag geordnet → der AE *lernt* die Muster, die
Schwäche liegt in der Score-Trennung. Output:
`figures/lstm_ae_reconstruction_<variant>_<wmz>.png`,
`figures/lstm_ae_latent_<variant>_<wmz>.png`,
`reports/lstm_ae_reconstruction_<variant>_<wmz>.csv`.
**Entscheidung 2026-07-14:** Die Thesis-Abbildungen (Abb. 18/19 in
MA.docx) bleiben auf dem Stand des vorhandenen GPU-Checkpoints
(Stage-7-Default-HPs) — die Diagnose ist qualitativ (Rekonstruktions-Güte,
Latent-Ordnung) und hängt nicht an den finalen HPO-Konfigurationen; ein
Re-Run nach der Score-Ablation ist nicht mehr vorgesehen.

**Score-Ablation** ([src/tools/lstm_ae_score_ablation.py](../src/tools/lstm_ae_score_ablation.py), ergänzt 2026-07-09).
Testet die aus der Rekonstruktions-Sicht abgeleitete Hypothese
**kausal**: Liegt die LSTM-AE-Schwäche in der Fehler→Score-Abbildung
statt in der Architektur? **Dasselbe trainierte Modell** wird unter
vier Scorings bewertet (`score_mode` in
[src/models/lstm_ae.py](../src/models/lstm_ae.py)):
`window_mse` (Stage-10-Baseline: MSE über Fenster × alle 18 Features —
mittelt eine punktuelle Abweichung über bis zu 432 Werte und
verwässert sie), `channel_mse` (nur Ziel-Kanal kw_mean/residual/trend),
`last_step_mse` (nur letzter Zeitschritt = punktschärfste Attribution)
und `mahalanobis` (Original-Scoring von EncDec-AD, Malhotra u. a. 2016:
Fehlervektor des letzten Zeitschritts gegen die auf den
**Trainings**-Daten geschätzte Fehlerverteilung N(μ, Σ) — berücksichtigt
Feature-Skalen und -Korrelationen; Abweichung zum Paper: Statistik auf
Train statt separatem Validierungs-Normalset, kein Leakage).
Evaluations-Protokoll = exakt Stage 10 (Re-Fit mit Stage-8-Best-HPs,
Schwelle = 0,99-Val-Quantil **je Modus**, point-adjusted P/R/F1 +
ROC-/PR-AUC auf eligiblen Test-Stunden); Hauptergebnisse bleiben
unangetastet (eigenes CSV `reports/lstm_ae_score_ablation.csv`,
Spalte `delta_f1_vs_window`). `--use-saved` = Smoke-Test auf
Stage-7-Checkpoints (Default-HPs, nicht für Thesis-Zahlen; verifiziert
2026-07-09 auf dem Mac/MPS). **Interpretation:** F1-Sprung unter einem
alternativen Scoring = Kriteriums-Mismatch kausal belegt
(„Score-Reparatur statt Architekturwechsel"); bleibt F1 niedrig, ist
die DL-Schwäche robuster belegt — beides verwertbar (Kap. 4 Diagnostik,
§ 5.6). **Durchgeführt 2026-07-20 auf der RTX 3080 Ti** (CUDA, Re-Fit
aller 9 Jobs mit Stage-8-Best-HPs, 8:08 min Wandzeit). Ergebnisse und
Interpretation: ergebnisbericht § 12.8.2 — Score-Reparatur in 3 von 9
Jobs empirisch belegt (Δ F1 bis +0,689 unter `last_step_mse` auf
residual/wmz_1), in 6 von 9 bleibt die LSTM-AE-Schwäche unter allen vier
Modi bestehen; in keinem Job wird der Klassik-Champion unter irgendeinem
Scoring geschlagen.

**Protokoll-Klassifikation der Top-50-Listen** ([src/tools/qualitative_klassifikation.py](../src/tools/qualitative_klassifikation.py), ergänzt 2026-07-10).
Operationalisiert das Protokoll der qualitativen Evaluation (Thesis
Abschnitt 3.5.2): Jede Stunde der Stage-10-Top-50-Listen
(`reports/stage10_qualitative_*.csv`) wird gegen die injizierte
Ground-Truth **beider** Schienen-Gruppen und die realen Fehler-Flags
klassifiziert (`TP:<typ>` / `andere-Schiene:<typ>` /
`realer-Datenfehler` / `no_data` / `Fehlalarm/offen`). Liefert die in
Kap. 4.6 der Thesis berichteten Fallstudien-Zahlen (A: Konstanz auf
raw/wmz_2 49/50 Plateau-TP; B: LOF auf residual/wmz_1 12 Spike-TP +
38 Drift-Stunden = Schienen-Übergriff; C: PELT auf trend/wmz_1 12/13
echt markierten Stunden in nicht-stationären/langen Ereignissen -
binäre Scorer werden auf die tatsächlichen Alarme gefiltert, da die
Top-50-Liste sonst mit Null-Score-Stunden aufgefüllt ist (Fix 2026-07-14); Champion-IForest: auf
raw/wmz_3 dominieren 22 real geflaggte Sensorfehler-Stunden die
Top-50). Output: `reports/qualitative_klassifikation.csv` (eine Zeile
je Job × Kategorie, alle 33 Jobs).

**Reihenfolge im Nacht-Workflow.** Seed-Varianz und Score-Verteilungs-Diagnose laufen **nach** dem
HPO-Lauf (Stage 8) und der Test-Eval (Stage 10), als Schritte 6 und 7
in HANDOFF.md § 3.5. Compute-Aufwand: jeweils unter 1 min für die
Seed-Varianz (reine Pandas-Aggregation), unter 5 min für die
Score-Verteilungen (18 Score-Aufrufe auf bereits trainierten Modellen).

**Quellen:** Seed-Varianz/Stochastik (Reimers und Gurevych 2017;
Bouthillier u. a. 2021); Score-Diagnose-Rahmen (Davis und Goadrich 2006;
Saito und Rehmsmeier 2015 — Imbalance-Sensitivität von ROC-/PR-Kurven).

---

## 8. Reproduzierbarkeit und neue Datensätze

### Projektstruktur

```
anomaly_detection_master_thesis/
├── data/
│   └── <dataset>/                  z.B. gebaeude_a/
│       ├── logging_heat-energy_*.csv
│       └── open-meteo-*.csv
├── outputs/
│   └── <dataset>/                  Output-Bereich pro Datensatz
│       ├── parquet/   (Stage 1–6 Tabellen + Stage-9 Injektion/Ground-Truth)
│       ├── csv/       (gleiche Daten als CSV)
│       ├── reports/   (Gap-/Glitch-Logs, Stage-10 Metriken + qualitativer Export)
│       ├── figures/   (Explorations-Plots)
│       ├── scalers/   (Stage-6 StandardScaler je Variante)
│       ├── models/    (Stage-7 Modelle: <variant>/<wmz>/<name>.pkl)
│       └── hpo/       (Stage-8 best_hparams.json + hpo_log.csv)
├── src/                            Pipeline-Code
│   ├── stage{1..10}_*.py           Pipeline-Stages
│   ├── anomaly_injection.py        Injektions-Bibliothek (Stage 8/9)
│   ├── injection_apply.py          Injektion + Re-Normalisierung (geteilt)
│   ├── evaluation.py               Metriken (point-adjust, ROC-AUC)
│   ├── result_io.py                nicht-destruktives Schreiben (Stage 8/10)
│   ├── models/                     Detektoren (base, zscore, lof, iforest, pelt, lstm_ae, registry)
│   └── tools/                      Hilfs-/Explorations-Skripte (data_sweep, lag_ablation, *_explore, macroevents)
├── docs/                           Methodik-Dokumentation
└── .venv/                          isolierte Python-Umgebung
```

### Pipeline ausführen

```powershell
# Einmalig pro Terminal: Virtual Env aktivieren
.\.venv\Scripts\Activate.ps1

# Komplette Pipeline auf Default-Datensatz (gebaeude_a):
python src\stage1_load.py
python src\stage2_preprocess.py
python src\tools\stage2_explore.py  # optional: Plots zu Stage 2
python src\stage3_stl.py
python src\tools\stage3_explore.py  # optional: Plots zur STL-Dekomposition
python src\stage4_features.py
python src\stage5_split.py
python src\stage6_normalize.py
python src\stage7_train.py
python src\stage8_hpo.py
python src\stage9_inject.py
python src\stage10_evaluate.py

# Schneller Probelauf bzw. ohne GPU (LSTM-AE auslassen):
python src\stage7_train.py    --max-train-rows 2000
python src\stage8_hpo.py      --models zscore lof iforest pelt
python src\stage10_evaluate.py --models zscore lof iforest pelt

# Diagnose-Werkzeug: Datenmengen-Sensitivitaet eines einzelnen
# Detektors auf einem (Variante x WMZ)-Job (Sektion 7.8).
python src\tools\data_sweep.py --variant raw --wmz wmz_1 --model iforest `
    --train-rows 2000 5000 10000 0
# 0 = alle verfuegbaren Train-Stunden. HPs werden default aus
# best_hparams.json geladen; Output kumuliert in
# outputs/<ds>/reports/data_sweep.csv.
```

### Weiteren Datensatz hinzufügen

1. Neuen Unterordner `data/<neuer_name>/` anlegen.
2. Die 5 WMZ-CSVs und die Open-Meteo-CSV mit denselben Dateinamen-Mustern dort
   ablegen.
3. Pipeline mit `--dataset`-Flag aufrufen:

```powershell
python src\stage1_load.py        --dataset <neuer_name>
python src\stage2_preprocess.py  --dataset <neuer_name>
python src\tools\stage2_explore.py --dataset <neuer_name>
python src\stage3_stl.py         --dataset <neuer_name>
python src\tools\stage3_explore.py --dataset <neuer_name>
python src\stage4_features.py    --dataset <neuer_name>
python src\stage5_split.py       --dataset <neuer_name>
python src\stage6_normalize.py   --dataset <neuer_name>
python src\stage7_train.py       --dataset <neuer_name>
```

Outputs landen automatisch in `outputs/<neuer_name>/`. Bestehende
Datensätze werden nicht überschrieben.

### Versionen und GPU-Setup

Python 3.12. Plattform-unabhängig gepinnte Pakete in
[requirements.txt](../requirements.txt) / [requirements-lock.txt](../requirements-lock.txt):
`pandas`, `pyarrow`, `matplotlib`, `statsmodels`, `holidays`,
`scikit-learn`, `ruptures`.

**PyTorch** (nur für den LSTM-Autoencoder) ist bewusst **nicht** im
Lock-File und wird pro Maschine separat installiert, weil der Build vom
Beschleuniger abhängt:

- **Trainings-Laptop (RTX 3080 Ti mobile, CUDA):**
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Apple Silicon (M4 Pro, MPS/Metal):** `pip install torch` mit einem
  **nativen arm64-Python**. Läuft das venv-Python unter Rosetta als
  x86_64 (Test: `python -c "import platform; print(platform.machine())"`
  → `x86_64` auf arm64-Hardware), zieht pip das veraltete x86-Wheel
  `torch==2.2.2`, das mit `numpy>=2` inkompatibel ist — dann venv mit
  nativem arm64-Python neu anlegen.
- **CPU-only:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`

Der LSTM-AE wählt das Gerät automatisch (`device="auto"` → CUDA vor MPS
vor CPU). **Determinismus:** Stages 1–6 und die klassischen Detektoren
(Z-Score, LOF, IF, PELT) sind deterministisch; ein erneuter Lauf
produziert bit-identische Outputs. Einzige Ausnahme ist der LSTM-AE auf
GPU (CUDA/MPS-Kernel nicht bit-identisch).

---

## 9. Literatur

Quellen, die die methodischen Entscheidungen der Pipeline begründen,
formatiert nach **DIN ISO 690** (Verweis-Datum-System). Die Inline-
Verweise in den Sektionen oben (Autor Jahr) lösen sich hier auf.

Quellenverwaltung mit **Zotero**: Die Datei
[docs/literatur.bib](literatur.bib) (BibTeX) kann direkt in Zotero
importiert werden (*Datei → Importieren → Datei wählen*). Für die
DIN-ISO-690-Formatierung den CSL-Stil „DIN ISO 690" in Zotero installieren
(*Einstellungen → Zitieren → Stile → „Weitere Stile abrufen"*, nach
„ISO 690" suchen) und im Zotero-Word-Plugin als Zitierstil wählen; Zotero
formatiert Zitate und Literaturverzeichnis dann automatisch. **Hinweis:**
Seitenzahlen von Konferenzbeiträgen (WWW 2018, AAAI 2022) sind in der
`.bib` bewusst leer gelassen und vor der Abgabe zu ergänzen; die
arXiv-IDs (`eprint`) dienen zur Verifikation.

**Dekomposition (Stage 3, §4)**

- CLEVELAND, Robert B., William S. CLEVELAND, Jean E. McRAE und Irma
  TERPENNING, 1990. STL: A Seasonal-Trend Decomposition Procedure Based on
  Loess. *Journal of Official Statistics*. 6(1), 3–73.
- BANDARA, Kasun, Rob J. HYNDMAN und Christoph BERGMEIR, 2021. MSTL: A
  Seasonal-Trend Decomposition Algorithm for Time Series with Multiple
  Seasonal Patterns [online]. *arXiv*:2107.13462. Verfügbar unter:
  https://arxiv.org/abs/2107.13462
- HOCHENBAUM, Jordan, Owen S. VALLIS und Arun KEJARIWAL, 2017. Automatic
  Anomaly Detection in the Cloud Via Statistical Learning [online].
  *arXiv*:1704.07706. Verfügbar unter: https://arxiv.org/abs/1704.07706 —
  saisonale Dekomposition mit anschließender Detektion auf dem Residuum
  (S-H-ESD); begründet die Residuum-Schiene (§4.1). Preprint; für die
  Stage-2-Fehlererkennung durch Vallis u. a. 2014 ersetzt.
- VALLIS, Owen, Jordan HOCHENBAUM und Arun KEJARIWAL, 2014. A Novel
  Technique for Long-Term Anomaly Detection in the Cloud. In: *6th USENIX
  Workshop on Hot Topics in Cloud Computing (HotCloud 14)*. Philadelphia,
  PA: USENIX Association. — Peer-reviewte, stückweise Median/MAD-Langzeit-
  Filterung; begründet die regelbasierte Stage-2-Fehlererkennung (§3,
  robuste Median-/Cummax-Statistik).

**Feature Engineering (Stage 4)**

- CHRIST, Maximilian, Nils BRAUN, Julius NEUFFER und Andreas W.
  KEMPA-LIEHR, 2018. Time Series FeatuRe Extraction on basis of Scalable
  Hypothesis tests (tsfresh – A Python package). *Neurocomputing*. 307,
  72–77. — rollierende Fenster-Statistiken als Zeitreihen-Merkmale.
- DEMIR, E. und S. GUNAL, 2025. Short-term electricity consumption
  forecasting with deep learning. *The Journal of Supercomputing*. 81. —
  domänennahes Anwendungsbeispiel: zyklische Kodierung von Stunde/
  Wochentag/Monat plus Wetter-Features (Temperatur, Feuchte) mit LSTM/CNN.

**Anomalie-Taxonomie und synthetische Injektion (§6, Stage 9)**

- CHANDOLA, Varun, Arindam BANERJEE und Vipin KUMAR, 2009. Anomaly
  Detection: A Survey. *ACM Computing Surveys*. 41(3), 1–58.
- BLÁZQUEZ-GARCÍA, Ane, Angel CONDE, Usue MORI und Jose A. LOZANO, 2021.
  A Review on Outlier/Anomaly Detection in Time Series Data. *ACM Computing
  Surveys*. 54(3), Artikel 56. — Review unüberwachter Zeitreihen-Anomalie-
  detektion, inkl. dekompositionsbasierter Verfahren.
- LAI, Kwei-Herng, Daochen ZHA, Junjie XU, Yue ZHAO, Guanchu WANG und Xia
  HU, 2021. Revisiting Time Series Outlier Detection: Definitions and
  Benchmarks. In: *Proceedings of the Neural Information Processing Systems
  Track on Datasets and Benchmarks (NeurIPS 2021)*.

**Split und Evaluation (Stage 5, Stage 10)**

- BERGMEIR, Christoph und José M. BENÍTEZ, 2012. On the use of
  cross-validation for time series predictor evaluation. *Information
  Sciences*. 191, 192–213.
- XU, Haowen, Wenxiao CHEN, Nengwen ZHAO u. a., 2018. Unsupervised Anomaly
  Detection via Variational Auto-Encoder for Seasonal KPIs in Web
  Applications. In: *Proceedings of the 2018 World Wide Web Conference
  (WWW '18)* [online]. arXiv:1802.03903.
- KIM, Siwon, Kukjin CHOI, Hyun-Soo CHOI, Byunghan LEE und Sungroh YOON,
  2022. Towards a Rigorous Evaluation of Time-series Anomaly Detection.
  In: *Proceedings of the AAAI Conference on Artificial Intelligence*
  [online]. 36(7). arXiv:2109.05257.

**Detektor-Modelle (Stage 7)**

- BREUNIG, Markus M., Hans-Peter KRIEGEL, Raymond T. NG und Jörg SANDER,
  2000. LOF: Identifying Density-Based Local Outliers. In: *Proceedings of
  the 2000 ACM SIGMOD International Conference on Management of Data*. New
  York: ACM, 93–104.
- LIU, Fei Tony, Kai Ming TING und Zhi-Hua ZHOU, 2008. Isolation Forest.
  In: *2008 Eighth IEEE International Conference on Data Mining (ICDM)*.
  Washington, DC: IEEE, 413–422.
- KILLICK, Rebecca, Paul FEARNHEAD und Idris A. ECKLEY, 2012. Optimal
  Detection of Changepoints with a Linear Computational Cost. *Journal of
  the American Statistical Association*. 107(500), 1590–1598.
- HOCHREITER, Sepp und Jürgen SCHMIDHUBER, 1997. Long Short-Term Memory.
  *Neural Computation*. 9(8), 1735–1780.
- MALHOTRA, Pankaj, Anusha RAMAKRISHNAN, Gaurangi ANAND, Lovekesh VIG,
  Puneet AGARWAL und Gautam SHROFF, 2016. LSTM-based Encoder-Decoder for
  Multi-sensor Anomaly Detection [online]. *ICML 2016 Anomaly Detection
  Workshop*. arXiv:1607.00148. Verfügbar unter:
  https://arxiv.org/abs/1607.00148

**Werkzeuge und Optimierung (Stage 6, Stage 8)**

- PEDREGOSA, Fabian u. a., 2011. Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research*. 12, 2825–2830.
- BERGSTRA, James und Yoshua BENGIO, 2012. Random Search for
  Hyper-Parameter Optimization. *Journal of Machine Learning Research*.
  13, 281–305.

**Datenexploration (`src/exploration/`, §5.4)**

- CLEVELAND, William S., 1979. Robust Locally Weighted Regression and
  Smoothing Scatterplots. *Journal of the American Statistical
  Association*. 74(368), 829–836. — foundational fuer LOWESS; begruendet
  die nicht-parametrische Glaettungslinie im Verbrauch-vs.-Temperatur-
  Scatter (Heizgrenz-Knick ohne parametrische Annahme).
- HAMMARSTEN, Stig, 1987. A Critical Appraisal of Energy-Signature
  Models. *Applied Energy*. 26(2), 97–110. — interpretativer Rahmen fuer
  den Heizgrenz-Knick (energy signature / balance point).
- TUKEY, John W., 1977. *Exploratory Data Analysis*. Reading, MA:
  Addison-Wesley. — Grundlagenwerk der EDA, begruendet die
  Wochenprofil-Heatmap (Stunde x Wochentag) als kompakte mehr-
  dimensionale Visualisierung.
- BOX, George E. P. und Gwilym M. JENKINS, 1976. *Time Series Analysis:
  Forecasting and Control*. Revised edition. San Francisco: Holden-Day.
  — Standardwerkzeug ACF/PACF zur empirischen Identifikation saisonaler
  Lags; begruendet die Wahl der MSTL-Perioden `[24, 168]`; ebenso die
  Kreuzkorrelationsfunktion (thermische Traegheit).
- VERBRUGGEN, Aviel, 1980. District heating: Estimation of a standard
  load duration curve. *International Journal of Energy Research*. 4(4),
  381–395. — Lastdauerlinie (Jahresdauerlinie) als Standard-Darstellung
  von Grundlast vs. Spitzenlast in Waermesystemen.
- FREDERIKSEN, Svend und Sven WERNER, 2013. *District Heating and
  Cooling*. Lund: Studentlitteratur. ISBN 9789144085302. — aktuelles
  Fernwaerme-Standardlehrbuch; Lastvariationen und Dauerlinien als
  Einordnung der WMZ-Lastprofile (moderner Beleg neben Verbruggen 1980).
- WANG, Xiaozhe, Kate SMITH und Rob J. HYNDMAN, 2006. Characteristic-Based
  Clustering for Time Series Data. *Data Mining and Knowledge Discovery*.
  13(3), 335–364. — varianzbasierte Trend-/Saison-Staerke-Masse
  `F = max(0, 1 − Var(R)/Var(K+R))`.
- KRASKOV, Alexander, Harald STÖGBAUER und Peter GRASSBERGER, 2004.
  Estimating mutual information. *Physical Review E*. 69(6), 066138. —
  k-Naechste-Nachbarn-Schaetzer der Mutual Information (Grundlage von
  `sklearn.feature_selection.mutual_info_regression`); erfasst die
  nicht-lineare Temperaturabhaengigkeit jenseits von Pearson-r.

**Datenmengen-Sensitivität (Daten-Sweep, §7.8)**

- CORTES, Corinna, L. D. JACKEL, Sara A. SOLLA, Vladimir VAPNIK und
  John S. DENKER, 1994. Learning Curves: Asymptotic Values and Rate of
  Convergence. In: *Advances in Neural Information Processing Systems
  (NIPS)*. 6, 327–334. — foundational zu Learning Curves.
- PERLICH, Claudia, Foster PROVOST und Jeffrey S. SIMONOFF, 2003. Tree
  Induction vs. Logistic Regression: A Learning-Curve Analysis. *Journal
  of Machine Learning Research*. 4, 211–255. — Learning-Curve-Analyse als
  Methodik (Kurven koennen sich kreuzen).
- VIERING, Tom und Marco LOOG, 2023. The Shape of Learning Curves: A
  Review. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
  45(6), 7799–7819. — umfassender Review (Form/Verlauf von Learning Curves,
  Effekt zusaetzlicher Trainingsdaten).

**Ausblick (§7.9)**

- XU, Jiehui, Haixu WU, Jianmin WANG und Mingsheng LONG, 2022. Anomaly
  Transformer: Time Series Anomaly Detection with Association Discrepancy.
  In: *International Conference on Learning Representations (ICLR 2022)*.
