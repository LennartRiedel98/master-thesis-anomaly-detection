# Gesamt-Ergebnisbericht — Pipeline Stages 1 bis 10

> **Hinweis zur veröffentlichten Fassung.** Dieser Bericht dokumentiert die
> Läufe auf den **realen Betriebsdaten**. Die liegen dem Repository nicht bei
> (siehe [README, Abschnitt „Daten"](../README.md#daten)), Gebäude, Betreiber,
> Standort und Zählernummern sind anonymisiert. Die verlinkten
> `outputs/gebaeude_a/…`-Artefakte entstehen erst bei einem Lauf auf diesen
> Daten und fehlen hier folglich. Wer den mitgelieferten Demo-Datensatz
> `demo_synthetic` durchlaufen lässt, erhält **andere Zahlen** — dessen Werte
> sind erfunden.
>
> Verweise auf `HANDOFF.md` und `MA.docx` betreffen das interne
> Arbeitsprotokoll bzw. die Arbeit selbst; beide sind nicht Teil dieses
> Repositories.

*Datensatz:* `gebaeude_a` (Gebäude A, 3 Wärmemengenzähler + Wetter)
*Zeitraum:* 2019-11-19 00:00 bis 2023-11-19 00:00 (exklusiv) = exakt 4 Jahre
*Frequenz:* nativ ~1-min, prozessiert stündlich → 35 064 Stunden
*Letzte Aktualisierung:* 2026-06-13 (Header dieses Berichts, nicht git-blame)
*Modelle (alle vollständig HPO + evaluiert):* Z-Score, LOF, IsolationForest,
PELT, LSTM-Autoencoder, Constancy (typ-spezifisch, nur Variante A; § 11.4.1)
*Berichtete Metriken:* point-adjusted P/R/F1 + schwellenfrei ROC-AUC **und
PR-AUC** (§ 11.1); LSTM-AE-PR-AUC aus den GPU-Reruns vom 2026-06-10
(§ 12.7.1 / § 12.8), Baseline-Zeilen s. § 11.2 †
*Hardware:* NVIDIA RTX 3080 Ti Mobile (16 GB VRAM), CUDA 12.4, torch 2.6.0
*HPO-Wandzeit (LSTM-AE):* ~3 h 30 min für 225 fits auf der RTX 3080 Ti Mobile
(deutlich schneller als die a-priori-Schätzung von 10–14 h, weil viele
Random-Search-Konfigurationen kleine `window`/`epochs`-Werte ziehen)

---

## 0 Pipeline-Überblick

| Stage | Aufgabe | Wichtigster Output |
|---|---|---|
| 1 | CSVs laden, mergen, auf 4-Jahres-Fenster clippen | `stage1_raw_merged.parquet` |
| 2 | Dual-Channel-Fehlererkennung, Bereinigung, Stunden-Aggregation | `stage2_hourly.parquet` + Flag-Logs |
| 3 | MSTL-Zerlegung (Trend + 24 h + 168 h + Residuum) | `stage3_stl.parquet` |
| 4 | Feature Engineering pro Detektions-Variante (A/B) | `stage4_features_{raw,residual}.parquet` |
| 5 | Zeitlich strikter Train/Val/Test-Split | `split_assignment.parquet` |
| 6 | Z-Standardisierung (StandardScaler, fit nur auf Train) | `stage6_normalized_<variant>.parquet` |
| 7 | Modelltraining mit Default-HPs | `models/<variant>/<wmz>/<modell>.pkl` |
| 8 | Hyperparameter-Optimierung auf injizierter Validierung | `hpo/best_hparams.json` |
| 9 | Synthetische Anomalie-Injektion ins Test-Set | `stage9_injected_*.parquet`, `stage9_ground_truth.parquet` |
| 10 | Finale Evaluation (P/R/F1/AUC + qualitative TopN + Realwelt-Check) | `reports/stage10_metrics.csv` u. a. |

Drei Detektions-**Varianten** mit unterschiedlicher Repräsentation des
Heizleistungs-Signals:

* **Variante A (raw):** stündlich gemittelte kW + volle Feature-Bibliothek
  (Rolling-Statistiken, zyklische Zeit-Encodings, Wetter, Feiertage)
* **Variante B (residual):** MSTL-Residuum als Hauptsignal + minimaler
  Kontext (Trend, Rolling-Std, `is_holiday`, Temperatur)
* **Variante C (trend):** ausschließlich der MSTL-Trend, gedacht für
  langsame Drifts und Strukturbrüche (PELT)

Drei Wärmemengenzähler (WMZ) je Variante: `wmz_1`, `wmz_2`, `wmz_3`.
Damit ergeben sich **30 (Variante × WMZ × Modell)-Jobs** pro Pipeline-
Lauf (3 + 3 + 1 Modelle × 3 WMZ × 3 Varianten, asymmetrisch weil Variante C
nur PELT und LSTM-AE bekommt). Ohne LSTM-AE bleiben **21 Jobs** übrig.

---

## 1 Stage 1 — Laden und Mergen

**Input:** fünf WMZ-CSV-Dateien (deutsches Format, `;`-getrennt, Komma als
Dezimal) und eine Open-Meteo-Wetter-CSV. Jede WMZ-Datei enthält pro Zähler
zwei Spalten: kumulierte Energie (MWh) und momentane Leistung (kW).
Sampling nativ ~1 Minute.

**Operationen:**
1. Pro Datei: Datum + Zeit zu DatetimeIndex zusammensetzen.
2. Meter-IDs (`10000001`, `10000002`, `10000003`) auf `wmz_1`, `wmz_2`,
   `wmz_3` umbenennen.
3. Outer-Join auf gemeinsamen Zeitstempel (Wetter ist stündlich → WMZ-
   Minuten-Slots haben NaN-Wetter).
4. Clipping auf `[2019-11-19 00:00, 2023-11-19 00:00)`.

**Quality-Report:**

| Kennzahl | Wert |
|---|---|
| Output-Shape | (2 117 880 Zeilen, 9 Spalten) |
| Zeitspanne | 2019-11-19 00:00 → 2023-11-18 23:59 |
| WMZ-Spalten | wmz_{1,2,3}_{mwh,kw} (6 Spalten) |
| Wetter-Spalten | temperature, humidity, precipitation (3 Spalten) |
| NaN-Anteil WMZ-Spalten | je 1,66 % (= 35 064 Stunden ohne Minutendaten) |
| NaN-Anteil Wetter-Spalten | je 98,34 % (Wetter nur 1×/h, Rest ist NaN) |

Die hohen Wetter-NaN-Werte sind kein Datenfehler, sondern eine direkte
Folge des Outer-Joins zwischen minuten- und stündlichen Reihen — sie
verschwinden mit der stündlichen Aggregation in Stage 2.

---

## 2 Stage 2 — Vorverarbeitung mit Dual-Channel-Fehlererkennung

Stage 2 ist methodisch das **Herzstück der Datenqualitäts-Pipeline** (siehe
Methodik 6.4 / 8.1). Sie erkennt Sensorfehler **gleichzeitig auf zwei
unabhängigen Kanälen** (kumulierte MWh + momentane kW) und reduziert das
Signal danach auf eine saubere stündliche Reihe.

### 2.1 Fehler-Typen (sechs Flags + Gap-Bedingung pro WMZ)

Die sieben Fehler-Indikatoren (sechs regelbasierte Flags je Kanal plus die
kanalübergreifende Gap-Bedingung) sind keine Übernahme einer fertigen
Typologie, sondern eine WMZ-spezifische Operationalisierung, die beide
Messkanäle des Wärmemengenzählers nutzt (kumulierte MWh, momentane kW).
Methodisch folgen die Detektoren dem etablierten Prinzip der robusten,
**median-basierten** Anomaliedetektion in Zeitreihen: Statt Mittelwert und
Standardabweichung — die durch die Ausreißer selbst verzerrt werden — dient ein
gleitender Median als ausreißerresistente Basislinie (Leys u. a. 2013; Vallis
u. a. 2014). Die Einordnung der erfassten Muster als Punkt- bzw.
Subsequenz-Anomalien und die Behandlung der Fehlerbereinigung als regulärer
Vorverarbeitungsschritt folgen der Übersicht von Blázquez-García u. a. (2021).

Die Abweichungsschwellen selbst sind nicht statistisch, sondern physikalisch
bzw. domänenbezogen motiviert und damit interpretierbar und auditierbar. Beim
**MWh-Kanal** markiert eine Sprungschwelle von 1,0 MWh eine bedeutsame
Glitch-Größe: Sie liegt um das Tausendfache über der Messauflösung von 1 kWh
(0,001 MWh) und damit weit jenseits von Quantisierungsrauschen. Beim
**kW-Kanal** wird ein Spike relativ zum lokal typischen Niveau definiert
(lokaler Median × Faktor 10) und durch einen festen Sockel (5 kW, in der
Größenordnung der zählerseitigen Grundlast) gegen Fehlalarme in Null-Phasen
abgesichert; eine zweite Bedingung (globales 95-Perzentil × 3) verlangt
zusätzlich globale Unplausibilität. Diese beiden frei gewählten Konstanten —
Faktor 10 und Sockel 5 kW — wurden auf ihre Ergebniswirkung geprüft
(Sensitivitätsanalyse, **Anhang B / § 18**): Die bereinigte Datenbasis ist
gegenüber dem Sockel praktisch invariant (≤ 14 von 35 064 Stunden über
0–10 kW) und gegenüber dem Faktor ab dem gewählten Wert 10 stabil (Knie der
Sensitivitätskurve; kleinere Faktoren über-flaggen die natürliche Volatilität
des Warmwasser-Zählers wmz_2). Zwei Flags beruhen schließlich ausschließlich
auf physikalischer Plausibilität bzw. Zähler-Logik und benötigen keine
statistische Grundlage: `is_kw_negative` (negative Leistung ist physikalisch
unmöglich) und `is_mwh_reset` (Rückfall des kumulierten Zählerstands unter sein
eigenes Maximum, erkannt per Cummax).

| Flag | Kanal | Bedeutung |
|---|---|---|
| `is_mwh_reset` | MWh | Kumulierter Wert fällt unter sein eigenes Maximum (Zählersprung) |
| `is_mwh_spike` | MWh | Unplausibler Sprung nach oben (> lokaler Median + 1 MWh) |
| `is_mwh_plateau` | MWh | Std ≈ 0 obwohl der kW-Kanal Aktivität zeigt (Sensor stuck) |
| `is_kw_negative` | kW | Physikalisch unmögliche negative Leistung |
| `is_kw_spike` | kW | > 10 × lokaler Median + globale Plausibilitätsschwelle |
| `is_kw_plateau` | kW | Std ≈ 0 obwohl die MWh-Reihe steigt (Sensor stuck) |
| `is_gap` | beide | Beide Signale gleichzeitig NaN |

**Cross-Channel-Awareness:** Plateaus werden nur geflaggt, wenn der jeweils
**andere** Kanal Aktivität zeigt. Damit werden echte Sommer-Off-Phasen nicht
fälschlich als Sensor-Plateau interpretiert.

### 2.2 Flag-Counts pro WMZ (über alle ~2,1 Mio. Minuten)

| WMZ | mwh_reset | mwh_spike | mwh_plateau | kw_neg | kw_spike | kw_plateau | Geflaggte Stunden | Interpoliert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wmz_1 | 6 303 | 0 | 43 | 0 | 2 | 517 | **5 563 (15,9 %)** | 5 563 |
| wmz_2 | 1 781 | 0 | 5 | 0 | 102 | 21 | **1 780 (5,1 %)** | 1 780 |
| wmz_3 | 157 287 | 105 | **416 031** | 2 648 | 0 | 1 611 | **18 225 (52,0 %)** | 13 519 |

**Lesart der wmz_3-Zahlen:** Über die Hälfte aller Stunden ist mindestens
einmal geflaggt — das Resultat der bereits dokumentierten ~18-Tage-Zero-
Spans (Methodik 5.3, vgl. Memory: `data-quality-wmz-glitches.md`).
Das hier ist also kein Pipeline-Fehler, sondern die korrekte Detektion
realer Sensorpathologie. Der Filter macht genau das, wofür er gebaut wurde.

### 2.3 kW-Sanity-Werte (stündliche Mittelwerte, NaN-bereinigt)

| WMZ | min | median | p99 | max | NaN-Stunden |
|---|---:|---:|---:|---:|---:|
| wmz_1 | 0,00 | 9,87 | 13,33 | 27,81 | 346 |
| wmz_2 | 0,00 | 4,86 | 53,76 | 78,92 | 346 |
| wmz_3 | 0,00 | 59,39 | 483,56 | 852,54 | 4 445 |

Die drei Zähler haben deutlich unterschiedliche Größenordnungen
(wmz_3 ist ein Sammler mit ~10× höherem Median). Das motiviert die
**WMZ-spezifischen Modelle** in den späteren Stages — ein gemeinsames
Modell über alle drei Zähler wäre ohne extreme Normalisierung instabil.

### 2.4 Bereinigung

Die Bereinigung läuft in zwei Schritten und **in genau dieser Reihenfolge**:
zuerst auf der nativen Minutenebene (Nullen + Interpolation), erst danach die
Aggregation auf das Stundenraster. Die Reihenfolge ist bewusst gewählt —
sub-stündliche Glitches werden entfernt, *bevor* die Stundenmittelung sie
verschmieren oder hinter dem Mittelwert kaschieren würde.

**Schritt 1 — Bereinigung auf Minutenebene:**

* **Nullen geflaggter Minuten:** Jede in §2.1 als fehlerhaft erkannte Minute
  wird im `kW`-Kanal auf `NaN` gesetzt. Bereinigt wird **ausschließlich das
  kW-Signal** (der spätere Modell-Input); die kumulierte MWh-Reihe dient nur
  der Detektion (dort haben die Fehler klarere Signaturen) und wird danach
  nicht weiter verwendet.
* **Interpolation kurzer Lücken (≤ 3 h):** Zusammenhängende NaN-Läufe bis zu
  drei Stunden werden linear interpoliert. Kurze Lücken lassen sich aus dem
  unmittelbaren Kontext plausibel überbrücken, ohne die Anomaliestruktur zu
  verfälschen; **längere Lücken bleiben bewusst `NaN`**, um keine Daten zu
  *erfinden*. Das `*_interpolated`-Flag protokolliert je Stunde, ob
  interpolierte Werte enthalten sind.

**Schritt 2 — Aggregation auf das Stundenraster:**

* **Stündliches Resampling per `mean(kW)`** auf den kanonischen, lückenlosen
  Stundenindex (35 064 Stunden über die vier Jahre).
* **Flag-Aggregation:** Die minütlichen Flags werden je Stunde zu Counts
  summiert; zusätzlich fasst `<wmz>_was_flagged` („irgendein Fehler in dieser
  Stunde") sie pro Zähler zu einem Boolean zusammen.

**Verwendung der Flags (wichtig für die Methodik):** Die Flags sind **kein
Modell-Input** — in einer Produktiv-Inferenz wären sie nicht verfügbar, und als
Feature würden sie die Fehler-Information ins Modell leaken. Sie dienen
ausschließlich als (a) **Trainings-Filter** (Stage 7 trainiert nur auf
ungeflaggten Stunden) und (b) **Ground-Truth der Kategorie A** (bekannte
Datenfehler) für die Evaluation.

**Output-Shape:** (35 064 Stunden, 32 Spalten) — drei bereinigte kW-Reihen +
Interpolations-Flags, Wetter sowie die aggregierten Fehler-Flags je Zähler.

### 2.5 Artefakte

* [stage2_hourly.parquet](../outputs/gebaeude_a/parquet/stage2_hourly.parquet)
* [stage2_flag_log.csv](../outputs/gebaeude_a/reports/stage2_flag_log.csv) — kompakter Flag-Log
  pro Stunde (nur Zeilen mit Flags)
* [stage2_gap_log.csv](../outputs/gebaeude_a/reports/stage2_gap_log.csv) — nur Stunden mit Gaps
* [stage2_glitch_log.csv](../outputs/gebaeude_a/reports/stage2_glitch_log.csv),
  [stage2_reset_log.csv](../outputs/gebaeude_a/reports/stage2_reset_log.csv) — Detailansichten

### 2.6 Cross-Meter-Korrelation der Lastprofile

Paarweise Korrelation der drei bereinigten stündlichen kW-Reihen
(Pearson und Spearman, über 30 619 paarweise vollständige Stunden;
Rangkorrelation robuster gegenüber der schiefen Lastverteilung):

| Paar | Pearson | Spearman |
|---|---:|---:|
| wmz_1 ↔ wmz_2 | 0,41 | 0,52 |
| wmz_1 ↔ wmz_3 | 0,47 | 0,55 |
| **wmz_2 ↔ wmz_3** | **0,74** | **0,63** |

**Lesart (physikalisch validiert, s. § 3.1):** Das stärkste Paar ist
**wmz_2 ↔ wmz_3** — beide tragen einen witterungs- bzw. nutzungsgetriebenen
Heizungsanteil (wmz_2 = TWW + dyn. Heizung Küche; wmz_3 = Heizung
Laborgebäude + Fußbodenheizung Küche), sodass Außentemperatur, Werktags-
und Essenszeit-Takt sie gemeinsam steuern. **wmz_1** (TWW Allgemeinbereiche)
korreliert nur schwach mit beiden (0,41 / 0,47): eine diffuse, kaum
witterungsabhängige Zapf-/Zirkulationslast mit eigener Dynamik.

**Methodische Konsequenz:** Die niedrigen Cross-Korrelationen bestätigen
zusammen mit den unterschiedlichen Größenordnungen (§ 2.3), dass die drei
Zähler *kein* gemeinsames Lastregime teilen — das rechtfertigt die
durchgängig **WMZ-spezifischen Modelle** und die strikte Feature-Trennung
je Zähler (Anti-Cross-Leakage, § 4.5). Ein Detektor, der von einem Zähler
auf einen anderen schließt, hätte hier keine tragfähige Grundlage.

### 2.7 Temperaturabhängigkeit — drei Maße im Vergleich

Die Kopplung an die Außentemperatur wird mit drei Maßen gemessen, die
**gestaffelt immer schwächere Annahmen** treffen:

| Maß | setzt voraus | erfasst |
|---|---|---|
| Pearson *r* | linearen Zusammenhang | Geraden-Trend der Werte |
| Spearman ρ | **monotonen** Zusammenhang | jede gleichsinnige Beziehung, auch gekrümmt |
| Transinformation (MI) | nichts | **beliebige** statistische Abhängigkeit |

MI wird mit dem kNN-Schätzer nach Kraskov, Stögbauer und Grassberger (2004)
bestimmt (Einheit *nats*, `random_state=0`). Da nats unnormiert sind, ist
zusätzlich das gaußsche Korrelationsäquivalent
*r*<sub>äquiv</sub> = √(1 − e<sup>−2·MI</sup>) angegeben — es macht die MI mit
den Korrelationskoeffizienten vergleichbar.

| WMZ | Pearson \|*r*\| | Spearman \|ρ\| | MI [nats] | *r*<sub>äquiv</sub> | Lag-Maximum |
|---|---:|---:|---:|---:|---:|
| wmz_1 | 0,360 | 0,463 | 0,177 | 0,546 | −0,454 (8 h) |
| wmz_2 | 0,370 | 0,428 | **0,338** | **0,701** | −0,444 (6 h) |
| wmz_3 | 0,704 | **0,862** | 0,737 | 0,878 | −0,731 (5 h) |

**Kernbefund 1 — die Staffelung ist bei allen drei Zählern gleich:**
|*r*| < |ρ| < *r*<sub>äquiv</sub>. Da jedes Maß genau eine Annahme mehr
fallen lässt, ist die Differenz zwischen zwei Maßen inhaltlich lesbar:

- **Pearson → Spearman** misst den Gewinn allein durch das Aufgeben von
  *Linearität*. Er ist bei **wmz_3 am größten** (0,704 → 0,862), passend zur
  gekrümmten, aber durchweg fallenden Heizkennlinie.
- **Spearman → MI** misst die Abhängigkeit **jenseits der Rangordnung**. Sie
  ist bei **wmz_2 am größten** (0,428 → 0,701).

**Kernbefund 2 — warum wmz_2 den größten Rang-Überschuss hat:** Dort steuert
die Temperatur nicht nur das *Niveau*, sondern auch die *Streuung*. Die
Standardabweichung je Temperaturklasse fällt von ≈ 24 kW im Frostbereich auf
< 3 kW oberhalb 18 °C (**Faktor 8**); bei wmz_3 sind es sogar Faktor 69
(158 → 2,3 kW), bei wmz_1 dagegen nur 1,14. Solche
**Regime-/Varianzabhängigkeit ist keine Rangbeziehung** und bleibt Spearman
daher verborgen — MI erfasst sie, weil sie die gesamte bedingte Verteilung
bewertet.

**Kernbefund 3 — der Knick an der Heizgrenze, beziffert:**

| WMZ | *r* bei T < 15 °C | *r* bei T ≥ 15 °C | Grundlast-Sockel |
|---|---:|---:|---:|
| wmz_1 | −0,288 | **−0,037** | ≈ 9,3 kW |
| wmz_2 | −0,289 | **−0,021** | ≈ 5,4 kW |
| wmz_3 | −0,589 | −0,301 | ≈ 0 kW |

Für die beiden Trinkwarmwasser-Kreise **verschwindet die
Temperaturabhängigkeit im Sommerregime praktisch vollständig**; übrig bleibt
eine rein nutzungsgetriebene Grundlast. Der über alle Stunden gemittelte
Pearson-Wert mischt beide Regime und unterschätzt die Kopplung im Heizbetrieb
entsprechend.

> **Präzisierung zur Monotonie:** Die Beziehung ist *nicht* richtungsumkehrend.
> Die Klassenmittel von wmz_3 fallen über alle 13 Temperaturklassen hinweg
> streng monoton (369 kW bei −12 °C → 0,04 kW bei +30 °C); wmz_1/wmz_2 zeigen
> am warmen Ende nur einen marginalen Wiederanstieg (< 0,3 kW, Rauschniveau).
> Die dominante Nichtlinearität ist ein **Knick mit Sättigung**, keine
> Umkehr — deshalb hilft Spearman hier sehr wohl (er schlägt Pearson bei
> allen drei Zählern) und ist nicht, wie bei echt nicht-monotonen Beziehungen,
> wirkungslos.

**Methodische Konsequenz:** Die Außentemperatur wird auch jenen Zählern als
Feature bereitgestellt, deren globale Pearson-Korrelation schwach wirkt
(wmz_1, wmz_2) — vorausgesetzt, das Modell kann Nicht-Linearitäten und
Regimewechsel abbilden (Isolation Forest, LSTM-AE). Für rein lineare bzw.
Score-basierte Verfahren bleibt der Zusatznutzen begrenzt.

**Einschränkung:** MI ist symmetrisch und richtungslos und belegt keine
Kausalität. Ein Teil der gemessenen Abhängigkeit kann von einem gemeinsamen
Treiber stammen (Jahreszeit steuert Temperatur *und* Nutzungsverhalten) —
vgl. die Confounder-Diskussion in der Arbeit.

Reproduktion: `python src/exploration/mutual_information.py` →
[mutual_information_temp.csv](../outputs/gebaeude_a/reports/mutual_information_temp.csv),
[mutual_information_temp.png](../outputs/gebaeude_a/figures/mutual_information_temp.png);
Lag-Kurven via `python src/exploration/temperature_crosscorrelation.py`.
Spearman, stückweise Korrelationen und Klassenstatistiken sind aus
`stage2_hourly.parquet` direkt reproduzierbar.

---

## 3 Stage 3 — MSTL-Zerlegung

**MSTL** (Multi-Seasonal-Trend Decomposition using LOESS) zerlegt jede der
drei kW-Reihen in vier additive Komponenten:

`signal = trend + seasonal_24h + seasonal_168h + residual`

* `trend` — langsam veränderliches Niveau (Jahresgang, Strukturbrüche)
* `seasonal_24h` — Tagesprofil (Nacht-/Tag-Last)
* `seasonal_168h` — Wochenprofil (Werktag vs. Wochenende)
* `residual` — was nicht durch Trend + Saison erklärt ist (Hauptsignal
  für Anomalie-Detektion in Variante B)

**Konfiguration: `periods=[24, 168]`, `robust=True`.** Drei
Entwurfsentscheidungen, jeweils mit Begründung:

* **Zwei Saisonalitäten (`periods=[24, 168]`).** Die Perioden sind in
  Stunden angegeben: **24 h** modelliert den Tagesgang (Tag-/Nacht-Profil
  des Wärmeverbrauchs), **168 h** (= 24 × 7) den Wochengang
  (Werktag/Wochenende). Beide werden gleichzeitig geschätzt; was nach
  Abzug von Trend + beiden Saisonkomponenten übrig bleibt, ist das
  **Residuum** — exakt die Eingangsgröße, auf der die residuumsbasierten
  Detektoren und Modelle (Variante B) Anomalien suchen.

* **Robuster Fit (`robust=True`).** LOESS-Glättungen lassen sich von
  Ausreißern „ziehen": Ohne Robustheit würde z. B. ein Verbrauchs-Spike
  den lokal geschätzten Trend bzw. die Saison nach oben verbiegen, sodass
  die Komponenten den Spike teilweise **absorbieren** — im Residuum wäre
  er danach kleiner und schlechter erkennbar. `robust=True` macht den Fit
  iterativ: Punkte mit großem Residuum erhalten in der nächsten Iteration
  weniger Gewicht (Down-Weighting). So bleibt die Anomalie im Residuum
  stehen, statt in Trend/Saison einzusickern — entscheidend, weil das
  Residuum gerade die Eingangsgröße der Anomalieerkennung ist
  (Selbstkontamination wird verhindert).

* **NaN-Handling — kein Training auf erfundenen Werten.** MSTL/LOESS
  benötigt eine lückenlose Reihe. Daher werden NaN-Positionen **für den
  Fit** linear interpoliert (Rechen-Krücke, damit die Zerlegung
  durchläuft), **im Residuum** aber wieder auf NaN zurückgesetzt. An
  ehemals fehlenden Stellen steht damit im Residuum kein synthetischer
  Wert, sondern eine Lücke; spätere Modelle (LSTM-AE etc.) und Detektoren
  trainieren dort nicht auf interpolierten Fantasiewerten und melden dort
  auch keine Scheinanomalie.

### 3.1 Strukturstärke der Komponenten (STL-Strength)

**Was die Kennzahl misst.** Für jede MSTL-Komponente wird ihre Stärke
*relativ zum Residuum* bestimmt — das zitierfähige Strength-of-Trend/
Seasonality-Maß nach Wang, Smith und Hyndman (2006):

`F_Trend  = max(0, 1 − Var(R) / Var(T + R))`
`F_Saison = max(0, 1 − Var(R) / Var(S + R))`

Jeder Wert liegt in [0, 1]; nahe 1 bedeutet, dass die Komponente die
Variabilität gegenüber dem Rauschen dominiert (berechnet in
`stl_strength.py`). Anders als ein roher Varianzanteil **konkurrieren die
Werte nicht miteinander** — jeder ist ein eigenständiges Signal-zu-Rausch-Maß
je Komponente. Der Wert ist damit ein **zeitreihen-struktureller
Fingerabdruck** jedes Zählers.

| WMZ | F_Trend | F_Saison (24 h) | F_Saison (168 h) |
|---|---:|---:|---:|
| wmz_1 | 0,61 | **0,30** | 0,25 |
| wmz_2 | 0,42 | **0,73** | 0,57 |
| wmz_3 | **0,82** | 0,66 | 0,55 |

**Was diese Werte sagen:**

* **wmz_3 = Raumheizung** (Laborgebäude + Fußbodenheizung Küche): stärkster
  Trend (0,82). Die Heizsaison — im Sommer komplett aus — erzeugt große,
  langsame Niveauwechsel, die als Trend dominieren; zugleich klar periodisch
  (Tages-Saison 0,66).
* **wmz_2 = Trinkwarmwasser + dynamische Küchenheizung**: stärkste
  **Tages-Saisonalität** (0,73, der höchste Wert im Datensatz) und
  ausgeprägter Wochengang (0,57) bei schwächerem Trend — Essenszeiten und
  Werktagsbetrieb der Küche prägen das ausgeprägte Tages-/Wochenprofil.
* **wmz_1 = Trinkwarmwasser der Allgemeinbereiche**: durchweg nur moderate
  bis schwache Strukturstärke (schwächste Saisonalität 0,30). Diffuse,
  sporadische Zapfungen und Zirkulations-/Grundlast statt scharfer
  Zapfspitzen erklären die schwache Saisonstruktur und den hohen Rauschanteil.

Die Strukturprofile decken sich damit **exakt mit den realen
Versorgungsfunktionen** der drei Wärmeübertrager (Betreiberauskunft des Wärmeversorgers):
trend-dominierte Raumheizung, stark tages-/wochenperiodische Küche+TWW,
schwach strukturiertes Allgemein-TWW. Das ist ein **physikalischer
Plausibilitäts-Beleg** der MSTL-Zerlegung (Aufgriff in § 14 / Diskussion).

**Methodische Konsequenz (warum diese Tabelle wichtig ist).** Jede
Komponente korrespondiert mit einer Detektions-**Schiene** der Pipeline,
sodass das Profil unmittelbar die geeignete Variante je Zähler vorhersagt:

* **Starke Saisonkomponente → Residuum-Variante (B).** Bei wmz_2 (Saison
  0,73 / 0,57) entfernt die MSTL-Zerlegung die vorhersagbare Struktur; das
  Residuum ist dann überwiegend „sauberes" Rauschen, vor dem stationäre
  Punkt-Anomalien deutlich hervortreten (Z-Score/LOF/IForest auf B).
* **Starke Trendkomponente → Trend-Variante (C).** wmz_3 ist mit F_Trend
  0,82 ausgeprägt nicht-stationär; Niveau-Sprünge und Drift sind hier die
  relevanten Anomalien und werden in der Trend-Schiene per PELT erfasst,
  nicht im Residuum.
* **Durchweg schwache Strukturstärke → schwierigster Fall.** wmz_1 zeigt in
  keiner Komponente eine hohe Stärke; entsprechend bleibt nach der Zerlegung
  viel unerklärte Varianz, in der Anomalien mit dem Eigenrauschen des
  Zählers konkurrieren — die Detektion ist dort grundsätzlich erschwert.

Damit liefert die Tabelle zugleich den empirischen Beleg für die im
Thesis-Titel adressierte **Heterogenität**: Die drei Zähler haben
grundverschiedene Strukturprofile (trend-dominiert / stark periodisch /
schwach strukturiert), weshalb **kein einzelner Detektor und keine einzelne
Variante für alle drei optimal sein kann** — die Begründung für die parallele
Detektor-Suite über drei Varianten (vgl. die Champion-Verteilung in § 11 und
die Hypothesenbewertung zu H1/H3).

> **Maß-Wahl für die MA:** In der Masterarbeit wird durchgängig dieses
> **STL-Strength-Maß** verwendet (Methodik § 3.2, Werte hier/§ 4.2;
> zitierfähig über Wang, Smith und Hyndman 2006). Stage 3 berechnet
> zusätzlich einen *rohen* Varianzanteil `Var(K)/Var(Signal)`
> (`stage3_stl.py:variance_share()`) — der ist nur ein **interner
> Sanity-Check** und geht **nicht** in die Arbeit ein (unpubliziert,
> summiert wegen Komponenten-Kovarianz nicht auf 1).

### 3.2 Residual-Statistiken (Hauptsignal für Variante B)

| WMZ | mean | std | min | max |
|---|---:|---:|---:|---:|
| wmz_1 | −0,02 | 0,74 | −12,2 | +18,0 |
| wmz_2 | +0,56 | 5,54 | −30,9 | +67,6 |
| wmz_3 | +1,08 | **40,74** | −298,4 | +381,6 |

Die Std des Residuums skaliert mit dem Lastniveau (wmz_3 etwa 55× wmz_1).
Das ist erwartet — entscheidend ist die Standardisierung in Stage 6, die
beide auf vergleichbare Skala bringt.

### 3.3 Visualisierungen

Erzeugt durch `stage3_explore.py`:

* [stl_trend_alle.png](../outputs/gebaeude_a/figures/stl_trend_alle.png) — Trends aller drei WMZ
* [stl_tagesprofil.png](../outputs/gebaeude_a/figures/stl_tagesprofil.png),
  [stl_wochenprofil.png](../outputs/gebaeude_a/figures/stl_wochenprofil.png) — Saisonalitäten
* [stl_residual_hist.png](../outputs/gebaeude_a/figures/stl_residual_hist.png) — Residuum-
  Verteilungen
* [stl_zoom_wmz_{1,2,3}.png](../outputs/gebaeude_a/figures/) — Detailausschnitte je Zähler

---

## 4 Stage 4 — Feature Engineering

Zwei Feature-Tabellen für die ersten beiden Detektions-Varianten:

### 4.1 Variante A (raw) — volle Feature-Bibliothek

Shape: **(35 064, 36)**. 30 Modell-Features + 6 Metadaten (was_flagged /
interpolated pro WMZ — kein Modell-Input, nur für Trainingsdaten-Filterung).

| Gruppe | Anzahl | Inhalt |
|---|---:|---|
| Signal | 3 | `<wmz>_kw_mean` |
| Rolling / Deviation | 15 | je WMZ: rolling_mean/std über 6 h und 24 h, Abweichung vom 24-h-Mittel |
| Zeit-Encodings | 10 | `hour_{sin,cos}`, `weekday_{sin,cos}`, `month_{sin,cos}`, `is_weekday`, `is_weekend`, `is_night`, `is_holiday` |
| Wetter | 2 | `temperature`, `humidity` |

### 4.2 Variante B (residual) — minimaler Kontext

Shape: **(35 064, 17)**. 11 Modell-Features + 6 Metadaten.

| Gruppe | Anzahl | Inhalt |
|---|---:|---|
| Signal | 3 | `<wmz>_residual` (aus Stage 3) |
| Trend (Kontext) | 3 | `<wmz>_trend` (Skalen-Kontext, falls Modell ihn nutzen will) |
| Rolling-Std | 3 | rolling_std_24h auf dem Residuum (Plateau-Detektor-Feature, Methodik H7) |
| Zeit | 1 | nur `is_holiday` (zyklische Encodings sind redundant zu MSTL) |
| Wetter | 1 | `temperature` (humidity hat schwache Korr. → weggelassen) |

**Begründung der unterschiedlichen Feature-Sätze:** Variante B testet
explizit Hypothese H3 der Methodik — dass die MSTL-Vorverarbeitung den
Modellen die Arbeit der Saisonalitäts-Erkennung abnimmt. Die zyklischen
Zeit-Encodings werden weggelassen, weil sie identische Information liefern
würden wie die bereits abgezogenen `seasonal_24h` / `seasonal_168h`.

### 4.3 Variante C (trend)

Wird aus Bequemlichkeit erst in Stage 6 direkt aus `stage3_stl.parquet`
gezogen (nur die drei `<wmz>_trend`-Spalten); kein eigenes Feature-File.

### 4.4 Feiertags-Sanity

Es gibt **984 Feiertagsstunden** (2,81 %) im 4-Jahres-Fenster, das
entspricht ca. **41 Feiertagen** für Berlin. Plausibel: ~10 gesetzliche
Feiertage × 4 Jahre = 40, plus 2024-Bewegung ≈ 41.

### 4.5 Feature-Slicing pro WMZ in der späteren Verwendung

In Stage 7/10 schneidet `model_features(df, variant, wmz)` für jedes WMZ
nur **seine eigenen** Signal/Rolling-Spalten heraus plus die geteilten
Features. Resultat:

| Variante | Features pro WMZ-Job | Zusammensetzung |
|---|---:|---|
| raw | 18 | 1 Signal + 5 Rolling/Dev + 10 Zeit + 2 Wetter |
| residual | 5 | 1 Resid + 1 Trend + 1 Rstd + 1 Holiday + 1 Temp |
| trend | 1 | 1 Trend (univariat) |

---

## 5 Stage 5 — Train/Val/Test-Split

**Zeitliche, strikte Aufteilung** (kein Shuffling, keine Random-Splits) —
identisch für alle Varianten und Modelle. Linksgeschlossene, rechtsoffene
Intervalle.

| Split | Start | Ende (inkl.) | Stunden | Anteil |
|---|---|---|---:|---:|
| Train | 2019-11-19 00:00 | 2022-10-31 23:00 | **25 872** | 73,8 % |
| Val | 2022-11-01 00:00 | 2023-04-30 23:00 | **4 344** | 12,4 % |
| Test | 2023-05-01 00:00 | 2023-11-18 23:00 | **4 848** | 13,8 % |

**Methodische Anmerkung (Methodik 7.2):** Das Test-Fenster fällt
absichtlich in das Jahr mit der schlechtesten Datenqualität — damit ist
es ein „realistischer" Stresstest und nicht durch Best-Case-Daten
geschönt. Die Stage-10-Eligibilitäts-Filter (`gt_no_data`,
`gt_known_sensor_issue`) entfernen daraus nur die Stunden, in denen
*gar kein* valides Signal vorliegt — der Rest bleibt im Test.

Output: [split_assignment.parquet](../outputs/gebaeude_a/parquet/split_assignment.parquet)
(eine Spalte `split` mit Categorical-Werten train/val/test).

---

## 6 Stage 6 — Normalisierung (StandardScaler)

Pro Variante wird ein eigener StandardScaler **ausschließlich auf
Train-Zeilen** gefittet und dann auf alle drei Splits angewendet. Mean und
Std werden in `outputs/<ds>/scalers/scaler_<variant>.parquet` persistiert.

| Variante | Output-Shape | Skalierte Spalten | Sanity Train: max\|mean\| | Sanity Train: max\|std−1\| |
|---|---|---:|---:|---:|
| raw | (35 064, 37) | 30 | 7,21 × 10⁻¹⁶ | 2,22 × 10⁻¹⁶ |
| residual | (35 064, 18) | 11 | 7,29 × 10⁻¹⁶ | 1,11 × 10⁻¹⁶ |
| trend | (35 064, 13) | 3 | 7,29 × 10⁻¹⁶ | 0,00 × 10⁰ |

Die Sanity-Werte liegen alle bei Maschinen-Epsilon — die Standardisierung
ist exakt. Das Output enthält neben den skalierten Features auch die
Metadaten (`*_was_flagged`, `*_interpolated`) und die Split-Spalte, damit
Stage 7 nur eine Datei lesen muss.

**Design-Detail (Methodik 7.3):** Die Verwendung von Population-Std
(`ddof=0`) und die Behandlung konstanter Spalten (std = 0 → std = 1)
sind exakt kompatibel mit `sklearn.StandardScaler`, sodass die Skalierung
unabhängig von der späteren Bibliothek reproduzierbar ist.

---

## 7 Stage 7 — Modelltraining mit Default-Hyperparametern

Iteriert über alle (Variante × WMZ × Modell)-Jobs aus
`models.registry.REGISTRY`, fittet jedes Modell mit Default-HPs und
persistiert es.

### 7.1 Trainings-Daten-Filterung (Methodik 6.4)

Für jedes WMZ werden Stunden mit aktiven Fehler-Flags
(`<wmz>_was_flagged == True`) **vor dem Training entfernt**. Damit lernen
die Modelle keine Datenfehler als „normales" Muster. Die Flags selbst
sind kein Modell-Input.

| WMZ | Train-Stunden (verfügbar) | Effektive Train-Stunden nach Flag-Filter |
|---|---:|---:|
| wmz_1 | 25 872 | **23 950** |
| wmz_2 | 25 872 | **24 414** |
| wmz_3 | 25 872 | **14 067** |

(Für Variante C/Trend wird konsistent gefiltert, obwohl PELT auf dem
geglätteten Trend läuft, in dem Punkt-Fehler ohnehin absorbiert sind —
einheitliche Filterung sichert die Vergleichbarkeit der Varianten.)

### 7.2 Trainings-Bilanz

| Variante | Jobs (ohne LSTM-AE) | Modelle |
|---|---:|---|
| raw | 12 | zscore, lof, iforest, **constancy** × 3 WMZ |
| residual | 9 | zscore, lof, iforest × 3 WMZ |
| trend | 3 | pelt × 3 WMZ |
| **Summe** | **24** | persistiert als 24 .pkl-Dateien (+ 9 LSTM-AE = 33 Jobs) |

Der **Constancy-Detektor** (typ-spezifisch, nur Variante A; § 11.4.1) ist
rein statistisch und CPU-only — er erweitert die raw-Schiene von 3 auf 4
klassische Modelle. Er wurde **als Reaktion auf die schlechte Plateau-
Erkennung der übrigen Detektoren** (Plateau-Recall 0, § 11.4.1) nachträglich
ergänzt, ist also kein Teil des ursprünglichen Modellsatzes.

Stage 7 nutzt **Default-Hyperparameter** der Modelle (z. B.
`n_neighbors=20` für LOF). Die per HPO optimierten HPs werden erst in
Stage 10 wieder eingesetzt — Stage 7 ist primär ein Smoke-Test der
Pipeline-Integrität (siehe Erklärung in Sektion 9.3).

---

## 8 HPO — was bedeutet das eigentlich?

**HPO** = **Hyperparameter-Optimierung**.

Jeder Anomalie-Detektor hat *Stellschrauben*, die **vor** dem Training
gesetzt werden müssen und nicht aus den Daten gelernt werden:

| Modell | Hyperparameter | Was er steuert |
|---|---|---|
| Z-Score | `aggregation` ∈ {max, l2, mean} | Wie der Score über mehrere Feature-Dimensionen reduziert wird |
| LOF | `n_neighbors` ∈ {5, 10, 20, 40, 80} | Größe der Nachbarschaft für die lokale Dichteschätzung |
| IsolationForest | `n_estimators`, `max_features` | Anzahl der Bäume, Feature-Subsampling-Rate |
| PELT | `penalty` ∈ {5, 10, 20, 30, 50, 75, 100} | Strafe pro zusätzlichem Change-Point |

**Bewusst NICHT getunt:** `contamination` (LOF / IsolationForest). Es
beeinflusst nur die interne `predict()`-Schwelle, nicht das Ranking von
`score_samples()`. Da unsere HPO rangbasierte ROC-AUC verwendet und die
finale Schwelle in Stage 10 datengetrieben (99 %-Quantil der Validierungs-
Scores) gesetzt wird, hätte `contamination` keinen Effekt.

### Wie hängen Training und HPO zusammen?

HPO **enthält** Training — nämlich `n` Mal. Pseudocode:

```python
for hp in grid:
    model = ModelClass(**hp).fit(X_train)   # <-- Training pro HP
    score = bewerte_auf_val(model)
    merke_dir(hp, score)
```

In unserer Pipeline sind Training und HPO trotzdem in **getrennte
Stages** ausgelagert:

| | Stage 7 | Stage 8 | Stage 10 |
|---|---|---|---|
| **Zweck** | Sanity / Default-Modelle bauen | Beste HPs finden | Finale Evaluation |
| **HPs** | Modell-Defaults | gesamtes Gitter | beste aus Stage 8 |
| **fit()-Aufrufe pro Job** | 1 | 3–9 (Klassik), 25 (LSTM) | 1 |
| **Output** | `.pkl` pro Job | `best_hparams.json` | Metriken / CSV |

Die `.pkl`-Dateien aus Stage 7 werden in der finalen Evaluation **nicht
weiterverwendet** — Stage 10 trainiert mit den besten HPs aus Stage 8
frisch. Die Trennung erlaubt aber, Stage 7 minutenschnell als
Smoke-Check laufen zu lassen und die `.pkl`s für Ad-hoc-Inferenz
(z. B. neue Datei einschicken und scoren) zu nutzen.

### Warum injizierter Validierungssatz?

Unsere Detektoren sind **unüberwacht** — auf den Rohdaten existieren
keine Labels. Wir injizieren daher synthetische Anomalien mit bekannter
Wahrheit (Methodik 7.5 / 7.6) und nutzen ROC-AUC (score-basierte Modelle)
bzw. point-adjusted F1 (PELT) als HPO-Zielgröße. Validierungs- und
Test-Seed sind **unterschiedlich** (Val = 8, Test = 99), damit die HPs
nicht auf die exakten Test-Anomalien getunt werden.

---

## 9 Stage 8 — Ergebnisse der HPO

### 9.1 Beste Hyperparameter pro Job

| Variante | WMZ | Modell | Beste HP | Val-Score |
|---|---|---|---|---:|
| raw | wmz_1 | zscore | aggregation=mean | 0,574 |
| raw | wmz_1 | lof | n_neighbors=10 | **0,729** |
| raw | wmz_1 | iforest | n_estimators=400, max_features=1,0 | 0,627 |
| raw | wmz_2 | zscore | aggregation=max | 0,507 |
| raw | wmz_2 | lof | n_neighbors=20 | **0,641** |
| raw | wmz_2 | iforest | n_estimators=100, max_features=0,5 | 0,559 |
| raw | wmz_3 | zscore | aggregation=max | 0,538 |
| raw | wmz_3 | lof | n_neighbors=40 | **0,581** |
| raw | wmz_3 | iforest | n_estimators=100, max_features=0,7 | 0,550 |
| residual | wmz_1 | zscore | aggregation=mean | 0,692 |
| residual | wmz_1 | lof | n_neighbors=80 | 0,702 |
| residual | wmz_1 | iforest | n_estimators=400, max_features=0,5 | **0,753** |
| residual | wmz_2 | zscore | aggregation=max | 0,528 |
| residual | wmz_2 | lof | n_neighbors=80 | 0,590 |
| residual | wmz_2 | iforest | n_estimators=200, max_features=0,5 | **0,613** |
| residual | wmz_3 | zscore | aggregation=mean | 0,526 |
| residual | wmz_3 | lof | n_neighbors=5 | 0,504 |
| residual | wmz_3 | iforest | n_estimators=400, max_features=0,5 | **0,625** |
| trend | wmz_1 | pelt | penalty=5,0 | 0,873 |
| trend | wmz_2 | pelt | penalty=10,0 | **0,999** |
| trend | wmz_3 | pelt | penalty=10,0 | 0,900 |
| raw | wmz_1 | constancy | window=4, baseline=168, sensitivity=0,5 | 0,940 |
| raw | wmz_2 | constancy | window=4, baseline=168, sensitivity=0,5 | 0,901 |
| raw | wmz_3 | constancy | window=4, baseline=72, sensitivity=0,5 | 0,940 |
| raw | wmz_1 | lstm_ae | win=24, hidden=32, layers=2, lr=0,01, ep=50 | 0,509 |
| raw | wmz_2 | lstm_ae | win=24, hidden=32, layers=2, lr=0,003, ep=30 | 0,533 |
| raw | wmz_3 | lstm_ae | win=24, hidden=32, layers=1, lr=0,003, ep=30 | 0,490 |
| residual | wmz_1 | lstm_ae | win=24, hidden=64, layers=2, lr=0,001, ep=100 | 0,611 |
| residual | wmz_2 | lstm_ae | win=24, hidden=16, layers=1, lr=0,001, ep=100 | 0,559 |
| residual | wmz_3 | lstm_ae | win=24, hidden=32, layers=2, lr=0,01, ep=50 | 0,467 |
| trend | wmz_1 | lstm_ae | win=24, hidden=32, layers=2, lr=0,0003, ep=100 | 0,524 |
| trend | wmz_2 | lstm_ae | win=48, hidden=16, layers=2, lr=0,01, ep=50 | **0,676** |
| trend | wmz_3 | lstm_ae | win=24, hidden=32, layers=1, lr=0,001, ep=100 | 0,640 |

Hinweis zum **LSTM-AE**: Er ist in **keiner** Variante × WMZ der beste
Detektor auf der Validierung (die Fett-Markierung fehlt in allen seinen
Zeilen). Sein bester Job ist `trend/wmz_2` (0,676); auffällig ist, dass
`window=24` dominiert (nur `trend/wmz_2` wählt 48) — die Random-Search-
Präferenz für das kurze Fenster ist damit belegt und die spätere
`window=48`-Hypothese aus dem Daten-Sweep widerlegt (§ 12.7 / § 13.1).

Hinweis: Der **Constancy**-Val-Score ist die **plateau-spezifische** ROC-AUC
(gegen die Plateau-only-Ground-Truth), nicht die Gesamt-Schienen-AUC wie bei
den übrigen score-basierten Modellen — der Detektor ist typ-spezifisch und
wird entsprechend selektiert (analog zur PELT-F1-Sonderbehandlung, § 11.4.1).

(Score = ROC-AUC für score-basierte Modelle, point-adjusted F1 für PELT.
Fett = bester Detektor je Variante × WMZ auf der Validierung.)

### 9.2 Beobachtungen zu den HPO-Mustern

* **LOF zeigt das deutlichste Optimum:** bei `raw/wmz_1` springt die
  AUC von 0,658 (k = 80) auf 0,729 (k = 10). Bei `residual/wmz_1` ist
  es genau andersherum (k = 80 ist am besten) — die residuale Schiene
  ist „glatter", damit definieren mehr Nachbarn die lokale Dichte
  verlässlicher.
* **IsolationForest reagiert vor allem auf `max_features`**, weniger auf
  `n_estimators`. Auf der residualen Schiene gewinnt aggressives Subsampling
  (`max_features = 0,5`), weil die fünf Residual-Features stark korreliert
  sind und Subsampling die Diversität der Bäume erhöht.
* **Z-Score hat fast keinen Spielraum** — die drei Aggregationen liegen
  innerhalb von ± 0,06. Das ist erwartet, weil Z-Score ein extrem simples
  Modell ist.
* **PELT zeigt einen klaren Knick bei penalty ≈ 10:** niedrige Strafe
  (5–10) liefert mehr Change-Points und erwischt damit fast alle
  injizierten Drift-/Strukturbruch-Anomalien. Ab penalty = 20 bricht
  der Score ein, weil PELT zu konservativ wird.

### 9.3 Voller HPO-Log

102 (Variante × WMZ × Modell × HP-Kombination)-Bewertungen unter
[hpo/hpo_log.csv](../outputs/gebaeude_a/hpo/hpo_log.csv). Beste HPs als JSON unter
[hpo/best_hparams.json](../outputs/gebaeude_a/hpo/best_hparams.json).

---

## 10 Stage 9 — Synthetische Anomalie-Injektion in das Test-Set

Mit Seed `99` (verschieden vom Val-Seed `8`) werden in das **Test-Fenster**
(2023-05-01 bis 2023-11-19, 4 848 Stunden) synthetische Anomalien
gemäß Methodik 6.3 / 7.6 eingestreut:

| WMZ | Stationäre B (Std.) | Nicht-stat. B (Std.) |
|---|---:|---:|
| wmz_1 | 1 016 | 3 558 |
| wmz_2 | 793 | 3 693 |
| wmz_3 | 1 073 | 3 417 |

Das Unterscheidungskriterium ist, ob die Anomalie das **Normalniveau dauerhaft
verschiebt** (Methodik § 6.3):

**Stationäre Typen:** `drop`, `leakage`, `plateau`, `spike` — *zeitlich
begrenzte* Auslenkung; das Signal kehrt danach zu seiner ursprünglichen
Verteilung zurück, das Normalniveau bleibt. Sichtbar in der Roh-/Residual-
Schiene (Trend reagiert kaum) → Varianten A/B. Ground-Truth: `gt_stat`.
**Nicht-stationäre Typen:** `drift`, `structural_break` — *persistente bzw.
graduelle* Niveau-/Regimeveränderung; das Normalniveau ändert sich dauerhaft.
Vom MSTL-Trend absorbiert, daher nur in der Trend-Schiene sichtbar → Variante C
(PELT). Ground-Truth: `gt_nonstat`.

Die volle Reihe (35 064 Stunden) wird gespeichert — nur die Test-Stunden
sind verändert, der Vor-Kontext bleibt sauber. Das ist relevant für
sequenzielle Modelle (LSTM-AE), die einen sauberen Aufwärm-Block brauchen.

**Ground-Truth-Spalten** (alle in [stage9_ground_truth.parquet](../outputs/gebaeude_a/parquet/stage9_ground_truth.parquet)):

* `gt_stat_<wmz>`, `gt_stat_label_<wmz>` — stationäre Anomalien
* `gt_nonstat_<wmz>`, `gt_nonstat_label_<wmz>` — nicht-stationäre
* `gt_known_sensor_issue_<wmz>` — Kategorie A (Stage-2-Flags)
* `gt_no_data_<wmz>` — kein valides Signal (in Eval ausschließen)
* `gt_known_regulatory` — Kategorie C (± 14 Tage um EnSikuMaV-Stichtag)

**Visualisierung der Injektion:** Die Injektions-Karte zeigt alle 144 Events
als Gantt-Zeitleiste über das Test-Fenster (Mai–Nov 2023), je WMZ ein Subplot,
Spuren pro Typ, gestrichelte Linie = Trennung stationär (oben) /
nicht-stationär (unten). Sie steht als Abbildung in der Arbeit (Kap. 3.5.1);
im Repo liegt sie nicht mehr, sondern wird erzeugt mit:

```bash
python src/tools/plot_injektionskarte.py            # -> outputs/<ds>/figures/
python src/tools/plot_injektionskarte.py --docs-copy  # zusaetzlich nach docs/
```

Eine tabellarische Fassung (alle Events mit Start, Ende und Dauer) lag früher
als `docs/injektions_karte.csv` im Repo. Sie wird von keinem Skript erzeugt,
sondern war ein Einmal-Export; die Angaben stehen vollständig in
`outputs/<ds>/parquet/stage9_ground_truth.parquet`.

---

## 11 Stage 10 — Evaluation auf dem Test-Set

### 11.1 Konfiguration

* **Schwellwert:** 99 %-Quantil der **sauberen** Validierungs-Scores
  (PELT: binär, also Schwelle = 0,5)
* **Eligible Stunden:** nur Stunden ohne `no_data` und ohne bereits
  bekannte Kategorie-A-Sensorfehler — damit der B-Wert nicht von
  Kategorie-A-Störungen verzerrt wird
* **Point-Adjustment:** F1 wird point-adjusted gemäß Methodik 10a
  berechnet (ein Event-Segment gilt als erkannt, wenn irgendein Punkt
  darin den Schwellwert überschreitet)
* **Doppelmetrik:** Berichtet werden sowohl die schwellenabhängige
  **point-adjusted P/R/F1** als auch die **schwellenfreien** Rang-Maße
  **ROC-AUC** und **PR-AUC** (Average Precision). Die PR-AUC ist bei stark
  unbalancierten Klassen (seltene Anomalien) aussagekräftiger als die
  ROC-AUC, weil sie auf die seltene Positiv-Klasse fokussiert, statt über
  beide Klassen zu mitteln (Davis und Goadrich 2006; Saito und Rehmsmeier
  2015). Beide AUC-Werte sind threshold-frei und damit robust gegen die
  in L5 (§ 15) diskutierte Point-Adjust-Überschätzung. Implementiert in
  `evaluation.py` (`roc_auc`, `pr_auc`), persistiert in
  [stage10_metrics.csv](../outputs/gebaeude_a/reports/stage10_metrics.csv)
  (Spalten `roc_auc`, `pr_auc`).

### 11.2 Gesamt-Metriken pro Job

**Übersicht — bestes Modell je Job** (F1, `stratum = overall`; LSTM-AE zum
Vergleich). Kompakte Konsolidierung der ausführlichen Tabelle darunter:

| Variante | WMZ | Bestes Modell | F1 | ROC-AUC | PR-AUC | LSTM-AE F1 |
|---|---|---|---:|---:|---:|---:|
| raw | wmz_1 | iforest | **0,911** | 0,836 | 0,565 | 0,212 |
| raw | wmz_2 | constancy | 0,773 | 0,485 | 0,225 | 0,005 |
| raw | wmz_3 | constancy | 0,022 | 0,480 | 0,229 | 0,006 |
| residual | wmz_1 | lof | **0,837** | 0,708 | 0,351 | 0,014 |
| residual | wmz_2 | lof | 0,322 | 0,738 | 0,306 | 0,005 |
| residual | wmz_3 | zscore | 0,019 | 0,578 | 0,265 | 0,006 |
| trend | wmz_1 | pelt | **0,700** | 0,500 | 0,732 | 0,611 |
| trend | wmz_2 | pelt | 0,531 | 0,499 | 0,762 | 0,001 |
| trend | wmz_3 | (alle 0,000)\* | 0,000 | 0,287 | – | 0,000 |

\* trend/wmz_3: alle Modelle F1 = 0 (Heizung im Sommer aus) — kein echter
„Sieger". Klassik-Champion je Schiene: **IForest** (raw/wmz_1), **LOF**
(residual), **PELT** (trend). Ausführliche Vollmetriken:

| Variante | WMZ | Modell | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| raw | wmz_1 | zscore | 0,769 | 0,763 | 0,766 | 0,787 | 0,400 |
| raw | wmz_1 | lof | 0,786 | 0,851 | 0,817 | 0,767 | 0,398 |
| raw | wmz_1 | **iforest** | 0,876 | 0,949 | **0,911** | **0,836** | **0,565** |
| raw | wmz_2 | zscore | 0,167 | 0,003 | 0,005 | 0,522 | 0,177 |
| raw | wmz_2 | lof | 0,647 | 0,189 | 0,292 | 0,570 | 0,191 |
| raw | wmz_2 | iforest | 1,000 | 0,003 | 0,005 | 0,550 | 0,201 |
| raw | wmz_3 | zscore | 1,000 | 0,003 | 0,006 | 0,528 | 0,259 |
| raw | wmz_3 | lof | 0,500 | 0,003 | 0,006 | 0,503 | 0,236 |
| raw | wmz_3 | iforest | 1,000 | 0,003 | 0,006 | 0,448 | 0,208 |
| residual | wmz_1 | zscore | 0,444 | 0,189 | 0,265 | 0,769 | 0,365 |
| residual | wmz_1 | **lof** | 0,865 | 0,811 | **0,837** | 0,708 | 0,351 |
| residual | wmz_1 | iforest | 0,908 | 0,315 | 0,468 | 0,815 | 0,476 |
| residual | wmz_2 | zscore | 0,087 | 0,005 | 0,010 | 0,530 | 0,182 |
| residual | wmz_2 | **lof** | 0,732 | 0,206 | **0,322** | 0,738 | 0,306 |
| residual | wmz_2 | iforest | 0,000 | 0,000 | 0,000 | 0,661 | 0,229 |
| residual | wmz_3 | zscore | 0,500 | 0,010 | 0,019 | 0,578 | 0,265 |
| residual | wmz_3 | lof | 0,500 | 0,003 | 0,006 | 0,457 | 0,218 |
| residual | wmz_3 | iforest | 0,500 | 0,003 | 0,006 | 0,353 | 0,177 |
| trend | wmz_1 | **pelt** | 0,997 | 0,539 | **0,700** | 0,500 | 0,732 \* |
| trend | wmz_2 | pelt | 0,997 | 0,362 | 0,531 | 0,499 | 0,762 \* |
| trend | wmz_3 | pelt | 0,000 | 0,000 | 0,000 | 0,500 | 0,720 \* |
| raw | wmz_1 | lstm_ae | 0,272 | 0,174 | 0,212 | 0,676 | n/a † |
| raw | wmz_2 | lstm_ae | 0,034 | 0,003 | 0,005 | 0,509 | n/a † |
| raw | wmz_3 | lstm_ae | 0,100 | 0,003 | 0,006 | 0,417 | n/a † |
| residual | wmz_1 | lstm_ae | 0,023 | 0,010 | 0,014 | 0,711 | n/a † |
| residual | wmz_2 | lstm_ae | 0,026 | 0,003 | 0,005 | 0,552 | n/a † |
| residual | wmz_3 | lstm_ae | 0,111 | 0,003 | 0,006 | 0,395 | n/a † |
| trend | wmz_1 | lstm_ae | 0,712 | 0,535 | 0,611 | 0,288 | n/a † |
| trend | wmz_2 | lstm_ae | 0,007 | 0,001 | 0,001 | 0,407 | n/a † |
| trend | wmz_3 | lstm_ae | 0,000 | 0,000 | 0,000 | 0,287 | n/a † |

\* **PELT-PR-AUC ist wie die ROC-AUC degeneriert** (binäre 0/1-Scores → die
Average-Precision kollabiert auf die Punkt-Präzision der gesetzten Flags und
ist *nicht* mit den kontinuierlichen Scores der übrigen Modelle vergleichbar).
Beurteilung von PELT allein über F1.
† **LSTM-AE-PR-AUC — siehe GPU-Reruns, nicht diese Baseline-Zeile.** Die hier
berichteten LSTM-Zeilen sind der Random-Search-Lauf auf der RTX 3080 Ti
(§ 12.3), der **vor** Einführung der `pr_auc`-Spalte lief. Same-hardware-PR-AUC
liegen aus den GPU-Reruns vom **2026-06-10** vor (§ 12.7.1 gridwh-GPU; § 12.8
Option E) — sie gehören aber zu **anderen** HP-Selektionen und werden daher dort
berichtet, nicht rückwirkend in diese Baseline-Zeilen gesetzt. Referenz
(bester Job `trend/wmz_1`): gridwh-GPU PR-AUC 0,622, Option E 0,624.

**Aggregierte F1 (Mittel über WMZ × Modell, alle 5 Modelle):**

| Variante | Ø F1 (Klassik) | Ø F1 (alle inkl. LSTM) | Ø ROC-AUC (alle) |
|---|---:|---:|---:|
| raw | 0,313 | 0,235 | 0,580 |
| residual | 0,215 | 0,164 | 0,602 |
| trend | 0,410 | 0,205 | 0,406 |

Die ROC-AUC = 0,500 für PELT auf trend ist erwartet: PELT liefert keine
kontinuierlichen Scores, sondern eine binäre 0/1-Reihe → AUC degeneriert
und ist nicht aussagekräftig. Hier zählt allein das F1. Für die LSTM-AE
auf trend liefert die AUC mit 0,288/0,407/0,287 sogar Werte deutlich
**unter** 0,5 — das Modell rangiert die Stunden in genau umgekehrter
Richtung als die Ground-Truth. Konzeptioneller Hintergrund: Drift- und
Strukturbruch-Anomalien führen zu einer langsameren Veränderung des
Trends, was der LSTM-AE als „normaler" rekonstruieren kann als die
abrupten saisonalen Übergänge — sein Rekonstruktions­fehler ist dann
*niedriger* in den Anomalie-Stunden, nicht höher.

### 11.3 Interpretation der zentralen Befunde

**(a) WMZ-1 dominiert.** Drei Detektoren erreichen auf `wmz_1` F1 ≥ 0,7
(IForest/raw 0,91, LOF/residual 0,84, LOF/raw 0,82, PELT/trend 0,70).
Auf `wmz_2` und `wmz_3` brechen die Metriken weitgehend ein. Ursache:
`wmz_1` hat das längste, sauberste Trainings-Fenster
(23 950 Stunden), während `wmz_3` durch die hohe Flag-Quote (52 %) und
die zugehörige Filterung nur **14 067 Trainings-Stunden** behält und
auch auf dem Test viele Stunden via `gt_no_data` / `gt_known_sensor_issue`
ausgeschlossen werden (307 eligible stationäre Stunden gegenüber
757 bei `wmz_2` und 625 bei `wmz_1`).

**(b) IsolationForest ist auf der Roh-Schiene unschlagbar.**
Bei `raw/wmz_1` erreicht IForest F1 = 0,911 — höchster Wert der gesamten
Studie. Auf der residualen Schiene gewinnt dagegen LOF (F1 = 0,837 bei
`wmz_1`). Lesart: die volle 18-dimensionale Rohrepräsentation
(inkl. Wetter und Zeitencodings) ist informativ genug, dass die
randomisierte Isolation einzelne Anomalien sauber abtrennen kann.
Die kompakte 5-dimensionale Residual-Schiene hat weniger Trennachsen
— hier hilft LOFs dichtebasiertes Vorgehen mehr.

**(c) Z-Score versagt fast überall.** Der Z-Score ist multivariat als
„parallele Aggregation univariater Detektoren" gebaut und kann
Kreuzfeature-Strukturen nicht erfassen. Akzeptable Werte erreicht er nur
dort, wo eine einzelne Feature-Dimension die Anomalie klar bezeugt
(wmz_1/raw, F1 = 0,77).

**(d) PELT erkennt Change-Points robust auf wmz_1 und wmz_2,
versagt komplett auf wmz_3.** F1 = 0,700 bzw. 0,531 vs. 0,000.
Die hohe Precision (≈ 0,997) zeigt, dass PELT mit penalty = 5–10 sparsam
mit Falschmeldungen umgeht. Auf `wmz_3` finden sich dennoch 44
Change-Points im Trend (siehe 11.5) — die niedrige Recall heißt also nicht,
dass PELT „nichts findet", sondern dass die gefundenen Change-Points nicht
zu den injizierten Drift-/Strukturbruch-Anomalien passen. Der
Sensor-Lückentrend von `wmz_3` produziert ohnehin viele echte
Strukturbrüche, die das Signal überlagern.

### 11.4 Recall pro Anomalietyp (`raw`, wmz_1 — bester Job)

**Übersicht — Recall je Anomalietyp × Modell** (Bestwert über die Jobs jeder
Schiene; „–" = Typ liegt nicht auf der Schiene des Modells). Das ist der
direkte **H1-Beleg (Methoden-Matching)**:

| Modell | spike | drop | plateau | leakage | drift | structural_break |
|---|---:|---:|---:|---:|---:|---:|
| zscore | 1,00 | 1,00 | 0,00 | 0,75 | – | – |
| lof | 1,00 | 0,12 | 0,25 | 0,88 | – | – |
| iforest | 1,00 | 0,67 | 0,00 | 1,00 | – | – |
| constancy | 0,00 | 0,00 | **1,00** | 0,88 | – | – |
| pelt | – | – | – | – | 0,50 | 0,33 |
| lstm_ae | 1,00 | 0,00 | 0,00 | 0,12 | 0,62 | 0,67 |

Kernaussagen: **spike** erkennen fast alle; **plateau** erkennt **nur der
Constancy-Detektor** (1,00) — alle anderen ≤ 0,25 (die Plateau-Lücke, § 11.4.1);
**drift/structural_break** sind reine Trend-Phänomene (nur PELT/LSTM-AE auf
Schiene C). Wichtig: hier steht **Recall**, nicht F1 — der LSTM-AE hat auf
drift/structural_break brauchbaren Recall (0,62/0,67), aber **schwache
Precision**, daher niedriges F1 (§ 12.4). Recall je Typ im Detail (bester Job
`raw`/wmz_1):

| Modell | drop | leakage | plateau | spike |
|---|---:|---:|---:|---:|
| zscore | 1,00 | 0,75 | 0,00 | 1,00 |
| lof | 0,00 | 0,88 | 0,00 | 1,00 |
| iforest | 0,67 | 1,00 | 0,00 | 1,00 |

**Spikes** werden von allen drei Modellen perfekt erfasst (Recall = 1,0).
**Leakage** und **Drop** sind solide detektierbar.
**Plateau** wird von **keinem** der fünf Standard-Modelle erkannt — konsistent
über alle WMZ und Varianten, **auch nicht vom LSTM-Autoencoder** (entgegen der
ursprünglichen Erwartung, dass das sequenzbasierte Modell diese Lücke füllt:
ein Rekonstruktions-AE rekonstruiert eine konstante Sequenz fehlerfrei und
flaggt sie deshalb gerade *nicht*). Ursache: ein Plateau ist eine *kollektive*
Anomalie — jeder Einzelwert ist plausibel, nur die **Konstanz** über mehrere
Stunden ist anomal; in den Z-Score-/Density-/Isolation-/Rekonstruktions-Räumen
entsteht kein Ausreißer. Diese Lücke wird durch einen typ-spezifischen Detektor
geschlossen (§ 11.4.1).

### 11.4.1 Plateau-Lücke geschlossen — Constancy-Detektor (Variante A)

Die fünf Standard-Detektoren (Z-Score, LOF, IForest, LSTM-AE auf raw und
residual) erreichen auf Plateaus durchweg **Recall 0** — eine kollektive
Anomalie ohne Punktausreißer, für die kein punkt-/rekonstruktionsbasiertes
Verfahren ausgelegt ist (§ 11.4). **Als direkte Reaktion auf diesen Befund**
wurde ein **typ-spezifischer `ConstancyDetector`** entwickelt und nachgerüstet
(nur Variante A; gerahmt unter **H1** =
Methoden-Matching, *keine* neue Hypothese). Er ist rein statistisch und
**CPU-only** (kein GPU). Score: `flatness × collapse` — momentane Konstanz
(`exp(−local_std/std_scale)`) mal **Varianz-Einbruch gegenüber der jüngsten
Eigen-Historie** des Zählers (`base_std` über ein langes, zurückversetztes
Kontextfenster). Damit operationalisiert er „konstant, *obwohl* aktiv erwartet"
über die Historie statt über ein absolutes Niveau; legitime Flachphasen waren
*schon* flach → kein Einbruch → kein Flag.

**Beste HPs (Stage 8, plateau-spezifische Selektion):** wmz_1/wmz_2
`{window 4, baseline 168, sensitivity 0,5}`, wmz_3 `{4, 72, 0,5}`.

| Recall pro Typ (raw) | drop | leakage | plateau | spike |
|---|---:|---:|---:|---:|
| constancy · wmz_1 | 0,00 | 0,13 | **1,00** | 0,00 |
| constancy · wmz_2 | 0,00 | 0,88 | **1,00** | 0,00 |
| constancy · wmz_3 | 0,00 | 0,13 | **0,00** | 0,00 |

| Plateau-Recall (raw) | wmz_1 | wmz_2 | wmz_3 |
|---|---:|---:|---:|
| **constancy** | **1,00** | **1,00** | 0,00 |
| zscore / lof / iforest / lstm_ae | 0,00 | 0,00 | 0,00 |

**Befund:** Die Plateau-Lücke ist auf den beiden **aktiven** Zählern geschlossen
(0 → 100 % Recall), während alle vier Bestandsdetektoren bei 0 bleiben. Overall
(gegen die gesamte stat. Schiene, daher erwartungsgemäß moderat, weil typ-
spezifisch): wmz_1 P 0,92 / R 0,18 / F1 0,30; wmz_2 P 0,68 / R 0,90 / F1 0,77.
Die **Top-50-Detektionen auf wmz_2 sind zu 49/50 Plateaus** — die stärksten
Scores sind sauber.

**Ehrliche Vorbehalte:**
- **wmz_3 = 0 ist korrekt, kein Versagen.** wmz_3 ist ein Heizungszähler, im
  Sommer faktisch aus (Jun–Aug ~3 kW vs. Winter ~200 kW). Die vier Test-
  Plateaus liegen in der saisonalen Null und halten ~0 kW → von der legitimen
  Abschaltung nicht trennbar; ein Flag wäre ein verkappter False Positive. Der
  Detektor trennt sauber „eingefroren **während aktiv**" (flaggen) von
  „konstant **weil saisonal aus**" (schweigen).
- **Leckage-Co-Detektion auf wmz_2 (0,88).** Auf dem varianzarmen Baseload-
  Zähler erscheinen die langgezogenen Leckage-Intervalle stellenweise als
  Niedrig-Varianz-Regime; die Punkt-Adjustierung schreibt dann das ganze
  Segment gut. Nebeneffekt, kein Designziel — drop/spike bleiben korrekt bei 0.
- **Tautologie-Vorbehalt.** Synthetische Plateaus sind perfekte Konstanten →
  der hohe Recall ist teils tautologisch; der eigentliche Beleg liegt in der
  **Precision** auf realen Flachphasen und der Prinzip-Demonstration.

Methodenfamilie: Inclán und Tiao 1994 + Killick u. a. 2012 (Varianz-Change-Point), Yeh u. a. 2016
(Matrix Profile), Lin u. a. 2007 (SAX). Reproduktion: `stage7→8→10
--variants raw --models constancy` (Mac-CPU, Sekunden).

### 11.5 Labelfreie Realwelt-Validierung — PELT vs. EnSikuMaV (10c)

Die Energiesicherungsmaßnahmenverordnung (**EnSikuMaV**) trat am
**2022-09-01** in Kraft (im *Trainings*-Fenster, daher kein injiziertes
Signal). Ein „guter" Trend-Detektor sollte unabhängig von synthetischen
Anomalien einen realen Change-Point in der Nähe dieses Datums liefern —
labelfreie Validierung mit der echten Welt.

| WMZ | Gefundene Change-Points (gesamt) | Im 30-Tage-Fenster um 2022-09-01 |
|---|---:|---|
| wmz_1 | 68 | 2022-08-29 19:00, 2022-09-13 09:00, 2022-09-20 11:00 |
| wmz_2 | 65 | 2022-08-31 01:00, 2022-09-17 13:00 |
| wmz_3 | 44 | 2022-09-15 11:00 |

**Alle drei WMZ liefern jeweils mindestens einen Change-Point innerhalb
± 30 Tagen um den EnSikuMaV-Stichtag** — bei `wmz_1` sogar einen drei Tage
vor dem offiziellen Inkrafttreten (typisch für Vorbereitungsmaßnahmen).
Dies ist ein **eigenständiger, nicht-synthetischer Beleg** für die
Funktionsfähigkeit der Trend-Schiene und besonders wichtig für die
Diskussion in der Thesis, weil er unabhängig von der Anomalie-Injektion
ist.

### 11.6 Qualitative Top-Detektionen (10b)

Für jede (Variante × WMZ × Modell)-Kombination sind die 50 Test-Stunden
mit dem höchsten Score in
[reports/stage10_qualitative_*.csv](.) exportiert — gedacht für die
manuelle TP/FP/Borderline-Klassifikation in der Thesis (Methodik 10b).

### 11.7 Seiten-Ablation — bringen Lag-Features einen Mehrwert? (ergebnisneutral)

Vor der Entscheidung, **keine** expliziten Lag-Features (`x_{t−1}`, `x_{t−24}`,
`x_{t−168}`) einzuführen (HANDOFF § 9.10), wurde der Mehrwert **empirisch**
geprüft — nicht nur strukturell argumentiert. Das Skript
[src/tools/lag_ablation.py](../src/tools/lag_ablation.py) vergleicht
IsolationForest auf `raw/wmz_1` **ohne vs. mit** Lag-Features bei *identischen*
besten HPs aus Stage 8, am **selben Lineal** (Schwelle = Val-q99, point-adjusted
F1 + Recall je Typ). Es ist **ergebnisneutral** — schreibt nichts nach
`outputs/`, tastet also den eingefrorenen Stage-10-Maßstab nicht an.

| IForest raw/wmz_1 | Precision | Recall | F1 | ROC-AUC | PR-AUC | Plateau-Recall |
|---|---:|---:|---:|---:|---:|---:|
| **ohne Lags** (18 Feat.) | 0,876 | 0,949 | **0,911** | 0,836 | 0,565 | **0/4** |
| **mit Lags** (21 Feat.) | 0,942 | 0,906 | **0,923** | 0,840 | 0,572 | **0/4** |
| **Δ** | +0,066 | −0,043 | **+0,012** | +0,004 | +0,007 | ±0 |

**Befund:** Lags bringen **keinen** belastbaren Mehrwert. Das +0,012 F1 liegt auf
Rausch-Niveau und ist reiner Precision/Recall-Tausch; per-Typ regredieren `drop`
(8/8 → 6/8) und `spike` (12/12 → 10/12) sogar leicht, `leakage` bleibt 8/8.
**Entscheidend: Plateau bleibt 0/4** — Lags schließen die Plateau-Lücke
**nicht** (ein Plateau ist eine *kollektive* Anomalie und wird von zeilenweisen
Lag-Features nicht erfasst). Damit ist die Designentscheidung „keine Lags,
stattdessen der typ-spezifische Constancy-Detektor (§ 11.4.1)" **empirisch
abgesichert**, nicht nur strukturell begründet.

**Ehrlicher Vorbehalt:** Single-Seed, nur `raw/wmz_1`, nicht nachgetunte HPs —
bei einer Baseline-F1 von 0,911 ist aber kein versteckter großer Gewinn zu
erwarten. Reproduktion: `python src/tools/lag_ablation.py` (Mac-CPU, Sekunden).

---

## 12 LSTM-Autoencoder — Vollständige HPO-Ergebnisse

Anhang A (§ 17) dokumentiert den **Smoke-Test** mit reduziertem Budget
(3 Random-Search-Configs, 5 000 Train-Stunden). Diese Sektion
dokumentiert den **vollen HPO-Lauf** (25 Random-Search-Configs aus
einem 216-Punkte-Gitter, volle Trainingsmenge), der am 2026-06-01/02
auf der RTX 3080 Ti Mobile gelaufen ist und die Smoke-Werte ablöst.

### 12.1 Reproduktions-Befehlsfolge (für künftige Datensätze)

```powershell
# torch + CUDA: einmalig pro Maschine (siehe § 17.1).

# Vor dem grossen Lauf snapshotten - Stage 8 schreibt best_hparams.json
# komplett neu, sonst gehen die Klassik-HPs verloren.
python src\merge_results.py --snapshot before_full_lstm

# Voller HPO-Lauf (25 Configs x 9 Jobs = 225 fits).
# Gemessene Dauer auf RTX 3080 Ti Mobile: 3 h 30 min.
python src\stage8_hpo.py --models lstm_ae

# Klassik-HPs wieder reinmischen (Snapshot als Basis, LSTM frisch).
python src\merge_results.py --merge before_full_lstm

# Test-Eval mit den vollen HPs (Snapshot + Eval + Merge analog).
python src\merge_results.py --snapshot before_full_lstm_eval
python src\stage10_evaluate.py --models lstm_ae
python src\merge_results.py --merge before_full_lstm_eval
```

Stage 9 muss **nicht** erneut laufen — das injizierte Test-Set ist
modellunabhängig und bereits persistiert.

### 12.2 Beste Hyperparameter nach vollständigem Random-Search

| Variante | WMZ | window | hidden | layers | lr | epochs | Val-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | wmz_1 | 24 | 32 | 2 | 0,01 | 50 | 0,509 |
| raw | wmz_2 | 24 | 32 | 2 | 0,003 | 30 | 0,533 |
| raw | wmz_3 | 24 | 32 | 1 | 0,003 | 30 | 0,490 |
| residual | wmz_1 | 24 | 64 | 2 | 0,001 | 100 | 0,611 |
| residual | wmz_2 | 24 | 16 | 1 | 0,001 | 100 | 0,559 |
| residual | wmz_3 | 24 | 32 | 2 | 0,01 | 50 | 0,467 |
| trend | wmz_1 | 24 | 32 | 2 | 0,0003 | 100 | 0,524 |
| trend | wmz_2 | 48 | 16 | 2 | 0,01 | 50 | 0,676 |
| trend | wmz_3 | 24 | 32 | 1 | 0,001 | 100 | 0,640 |

**Auffällig:** `window=24` dominiert (7 von 9 Jobs). Das ist die kleinste
Option des Suchraums {24, 48, 72}. Im Daten-Sweep (§ 13) schnitt aber
`window=48` auf `trend/wmz_1` beim **Test-F1** besser ab (0,776 vs. 0,611).
Daraus wurde zunächst vermutet, das Random-Search-Budget (25 von 216) sei
zu klein gewesen. **Diese Vermutung wurde mit HPO-Option A geprüft und
widerlegt** (§ 12.7): Auch im erschöpfenden `window×hidden`-Grid wird in
8 von 9 Jobs `window=24` gewählt; das eigentliche Problem ist kein zu
kleines Budget, sondern ein **Kriteriums-Mismatch** (Validierungs-ROC-AUC
unterscheidet die Fenstergrößen kaum, während `window=48` ein besseres
Test-F1 hätte).

### 12.3 Metriken auf dem Test-Set (volle HPs)

(Identisch zu den Stage-10-Zeilen für `lstm_ae` in § 11.2. **PR-AUC nicht
ausgewiesen** — dieser Random-Search-Lauf datiert vor Einführung der
`pr_auc`-Spalte in `evaluate()`. Same-hardware-PR-AUC liefern stattdessen die
GPU-Reruns vom 2026-06-10, § 12.7.1 / § 12.8.)

| Variante | WMZ | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| raw | wmz_1 | 0,272 | 0,174 | **0,212** | 0,676 |
| raw | wmz_2 | 0,034 | 0,003 | 0,005 | 0,509 |
| raw | wmz_3 | 0,100 | 0,003 | 0,006 | 0,417 |
| residual | wmz_1 | 0,023 | 0,010 | 0,014 | **0,711** |
| residual | wmz_2 | 0,026 | 0,003 | 0,005 | 0,552 |
| residual | wmz_3 | 0,111 | 0,003 | 0,006 | 0,395 |
| trend | wmz_1 | 0,712 | 0,535 | **0,611** | 0,288 |
| trend | wmz_2 | 0,007 | 0,001 | 0,001 | 0,407 |
| trend | wmz_3 | 0,000 | 0,000 | 0,000 | 0,287 |

### 12.4 Vergleich zum jeweiligen Klassik-Champion

| Variante | WMZ | Klassik-Champion | LSTM-AE | Δ F1 (LSTM − Klassik) |
|---|---|---|---|---:|
| raw | wmz_1 | iforest F1 = 0,911 | 0,212 | **−0,699** |
| raw | wmz_2 | lof F1 = 0,292 | 0,005 | −0,287 |
| raw | wmz_3 | (alle ≤ 0,006) | 0,006 | ±0 |
| residual | wmz_1 | lof F1 = 0,837 | 0,014 | **−0,823** |
| residual | wmz_2 | lof F1 = 0,322 | 0,005 | −0,317 |
| residual | wmz_3 | zscore F1 = 0,019 | 0,006 | −0,013 |
| trend | wmz_1 | pelt F1 = 0,700 | 0,611 | **−0,089** |
| trend | wmz_2 | pelt F1 = 0,531 | 0,001 | −0,530 |
| trend | wmz_3 | (pelt 0,000) | 0,000 | ±0 |

**LSTM-AE schlägt in keinem Setup den Klassik-Champion.** Am
nächstgelegenen ist `trend/wmz_1` (Δ −0,089) — also genau das Setup, wo
auch der Daten-Sweep den Mehrwert bereits angedeutet hat.

### 12.4.1 Ist der LSTM-AE der beste *Generalist*?

Naheliegender Gegeneinwand: Der LSTM-AE ist zwar nirgends Champion, ist er
aber vielleicht — als **einziges Modell, das auf allen drei Schienen (A/B/C)
läuft** — insgesamt die beste Wahl, wenn man nur *ein* Modell einsetzen will?
Die Klassiker sind Spezialisten (Z-Score/LOF/IForest nur raw/residual, PELT
nur trend, Constancy nur raw). Antwort: **Nein — er ist nicht der beste,
sondern der schwächste Allrounder.**

**Fairer Vergleich auf gleicher Job-Menge** (Ø F1, `stratum = overall`):

| Schiene | LSTM-AE | bester Klassiker auf derselben Menge |
|---|---:|---|
| raw + residual (6 Jobs) | 0,041 | LOF 0,380 (≈ 9× besser) |
| trend (3 Jobs) | 0,204 | PELT 0,410 (≈ 2× besser) |

**Rang des LSTM-AE je Job:** in **6 von 9 Jobs letzter Platz**; in den beiden
Trend-Jobs mit Konkurrenz Rang 2/2; der „Rang 1" in `trend/wmz_3` ist ein
Schein-Sieg (beide Modelle F1 = 0,000, Sommer-aus-Heizung). Einziger echt
konkurrenzfähiger Fall bleibt `trend/wmz_1` (0,611 vs. PELT 0,700).

**Ein-Modell-für-alles:** Über alle 9 Jobs erreicht der LSTM-AE Ø F1 = **0,096**;
die variantenspezifische Klassik (Job-Champion je Job) erreicht Ø F1 = **0,457**
— ein universelles LSTM-AE-Deployment wäre also um **Faktor ~5 schlechter**.

Das ist **kein Schwellwert-Artefakt**: die threshold-freie ROC-AUC liegt im
Mittel bei **0,471 < 0,5** (im Schnitt schlechter als Zufall — die
Score-Inversion aus § 12.7). Der einzige strukturelle Vorteil des LSTM-AE
(Universalität über alle Schienen) realisiert sich empirisch nicht; er stärkt
damit **H2** (Klassik schlägt DL) zusätzlich. Ehrliche Nuance: sein Potenzial
ist nicht null — auf `trend/wmz_1` deutet der Daten-Sweep mit `window = 48` +
mehr Daten auf F1 ≈ 0,776 (§ 13) —, es realisiert sich aber nur in dieser einen
Nische.

### 12.5 Recall pro Anomalietyp (LSTM-AE, jeweils wmz_1)

*raw / wmz_1:*

| Typ | drop | leakage | plateau | spike |
|---|---:|---:|---:|---:|
| LSTM-AE Recall | 0,000 (0/6) | 0,125 (1/8) | **0,000 (0/4)** | **1,000 (6/6)** |

*residual / wmz_1:*

| Typ | drop | leakage | plateau | spike |
|---|---:|---:|---:|---:|
| LSTM-AE Recall | 0,000 | 0,000 | **0,000** | **1,000** |

*trend / wmz_1:*

| Typ | drift | structural_break |
|---|---:|---:|
| LSTM-AE Recall | 0,625 (5/8) | 0,667 (2/3) |
| zum Vergleich PELT | 0,500 (4/8) | 0,333 (1/3) |

**Bemerkenswert auf der Trend-Schiene:** Auf Typebene erkennt der
LSTM-AE **mehr** Drift- und Strukturbruch-Events als PELT (drift 5/8 vs.
4/8, structural_break 2/3 vs. 1/3). Dass die Gesamt-F1 trotzdem
schlechter ist (0,611 vs. 0,700) liegt am Point-Adjust + an PELTs
extrem hoher Precision (0,997 vs. 0,712): PELT macht viel weniger
False Positives. Methodische Folgerung: PELT und LSTM-AE sind
**komplementär** — eine Ensemble-Variante könnte beide Stärken
verbinden.

### 12.6 Hypothesen-Bilanz (gegenüber den Erwartungen aus § 17)

**Gesamtbilanz der Thesis-Hypothesen H1–H5** (Zuordnung zu den Befunden):

| Hypothese | Kernaussage | Status | Kurzbegründung | Beleg |
|---|---|---|---|---|
| **H1** | Methoden-Matching (Verfahren ↔ Anomalietyp) | **bestätigt** | Jeder Typ wird vom passenden Verfahren erfasst; die Plateau-Lücke (Recall 0 bei allen 5 Standardmodellen) + der Constancy-Detektor, der sie schließt (1,0 / 1,0 / 0,0), sind selbst H1-Evidenz | § 11.4 / § 11.4.1 |
| **H2** | Klassik ≥ Deep Learning | **bestätigt (pro Klassik)** | LSTM-AE in 0/9 Jobs Champion; auch als Generalist schwächstes Modell (Ø ROC-AUC 0,471 < 0,5, Score-Inversion) | § 12.4 / § 12.4.1 |
| **H3** | MSTL-Residuum/Trend bringt Mehrwert | **überwiegend widerlegt** | Bestwert auf **Roh** (IForest 0,911 > LOF-Residuum 0,837); MSTL kein pauschaler Mehrwert | § 11.2 |
| **H4** | PELT > LSTM-AE bei Strukturbrüchen | **bestätigt** | PELT höhere Gesamt-F1 + Precision auf Trend; EnSikuMaV-Realbeleg. (Nuance: auf Typebene erkennt LSTM-AE mehr Events, aber mit schwacher Precision) | § 11.5 / § 12.5 |
| **H5** | Auflösung 1 min vs. 1 h | **analytisch beantwortet** | Stündlich angemessen (thermisch träge B-Anomalien; sub-stündliche Ereignisse sind Sensorfehler der Kat. A) | § 15-L7 |

Die folgende Tabelle ist die **LSTM-AE-spezifische** Detail-Bilanz gegenüber den
a-priori-Erwartungen aus § 17:

| Hypothese | Erwartung | Befund (volle HPs) | Status |
|---|---|---|---|
| Plateau-Recall > 0,5 | LSTM-AE schließt die Plateau-Lücke | 0,000 auf allen Schienen | **widerlegt** |
| residual/wmz_1 F1 ≳ LOF (0,837) | komplementär oder besser | 0,014 | **klar widerlegt** |
| trend/wmz_1 F1 ≈ PELT (0,700) | vergleichbar | 0,611 | **knapp verfehlt**, auf Typebene jedoch besser |

**Was die Diskussion in der Thesis braucht:** Eine ehrliche Einordnung
des LSTM-AE-Ergebnisses. Zwei plausible Erklärungen:

1. **Methodischer Effekt — Kriteriums-Mismatch (geprüft UND nachverfolgt,
   § 12.7 / § 12.8):** Der ursprüngliche Verdacht „Random-Search-Budget zu
   klein" wurde mit dem erschöpfenden `window×hidden`-Grid (HPO-Option A)
   **widerlegt** — `window=24` wird auch exhaustiv in 8/9 Jobs gewählt. Der
   Sweep-Befund (`window=48` besser im Test-F1) erklärt sich dadurch, dass die
   Validierungs-ROC-AUC die Fenstergrößen kaum trennt (Werte ~0,50–0,52). Der
   daraus abgeleitete „zielführende Hebel" — ein test-näheres **PR-AUC-Kriterium**
   — wurde mit **Option E (§ 12.8) auf der RTX getestet und brachte ebenfalls
   keinen Gewinn**: Die HPO wählt weiter `window=24`, `trend/wmz_1` bleibt bei
   F1 ≈ 0,61. Damit ist die Erklärung „nur ein Methoden-Artefakt" **erschöpfend
   ausgeschlossen** — es bleibt Erklärung 2.
2. **Strukturelle Eigenschaft der Daten:** Das Gebäude A hat eine
   stark periodische Last (Tagesprofil + Wochenprofil + Jahresgang). Die
   MSTL-Vorverarbeitung in Variante B nimmt diese Struktur exakt heraus
   und liefert dem LOF ein „weisseres" Residuum als Eingabe — das ist
   genau die Repräsentation, in der dichtebasierte Verfahren mit
   wenigen Trainingspunkten gut funktionieren. Der LSTM-AE muss die
   Struktur dagegen implizit lernen, was bei nur 24 000 Stunden Train
   und ~30 k Parametern an die Datengrenze stößt.

Nach den GPU-Reruns (§ 12.7.1 / § 12.8) ist Erklärung 1 (methodisches Artefakt)
ausgeschlossen; **es bleibt Erklärung 2 (strukturell)**: Auf diesem einfachen,
periodischen Regime ist der LSTM-AE-Nachteil real, nicht durch besseres HPO
heilbar. Das ist die belastbare Aussage für die Thesis — KI-Mehrwert ist erst
bei heterogeneren/gekoppelten Daten zu erwarten (§ 15/L3, Ausblick).

### 12.7 HPO-Option A — erschöpfender `window × hidden`-Grid (gridwh)

**Frage (aus § 12.2 / § 12.6):** War die Random-Search-Präferenz für
`window=24` ein Artefakt des kleinen Budgets (25 von 216 Konfigurationen)?
Der Daten-Sweep (§ 13) hatte mit `window=48` auf `trend/wmz_1` ein
besseres Test-F1 (0,776) angedeutet.

**Vorgehen:** `--lstm-strategy gridwh` durchsucht je Job **erschöpfend**
das Gitter `window ∈ {24, 48, 72} × hidden ∈ {16, 32, 64}` (9 Configs),
die übrigen HPs fix auf empirischen Defaults (`n_layers=2`, `lr=1e-3`,
`epochs=50`). 9 Jobs × 9 Configs = **81 Fits**. Gerechnet auf der **CPU des
M4-Pro-Macs** (MPS ist für `nn.LSTM` in PyTorch ≤ 2.12 wegen eines
Kernel-Speicherlecks unbrauchbar), in `float32`, neun Jobs **parallel**
(isolierte Temp-Datasets, Single-Merge danach) — Wandzeit **~60 min**
(sequentiell wären es ~12–15 h gewesen).

**Gewählte Konfiguration je Job (Auswahl nach Validierungs-ROC-AUC):**

| Variante | WMZ | window | hidden | Val-AUC | Test-F1 | Test-AUC |
|---|---|---:|---:|---:|---:|---:|
| raw | wmz_1 | 24 | 64 | 0,487 | 0,013 | 0,701 |
| raw | wmz_2 | 24 | 64 | 0,516 | 0,005 | 0,487 |
| raw | wmz_3 | 72 | 16 | 0,550 | 0,000 | 0,404 |
| residual | wmz_1 | 24 | 16 | 0,626 | **0,512** | 0,749 |
| residual | wmz_2 | 24 | 16 | 0,565 | 0,005 | 0,557 |
| residual | wmz_3 | 24 | 16 | 0,475 | 0,006 | 0,366 |
| trend | wmz_1 | 24 | 16 | 0,522 | **0,620** | 0,368 |
| trend | wmz_2 | 24 | 32 | 0,636 | 0,001 | 0,587 |
| trend | wmz_3 | 24 | 64 | 0,584 | 0,000 | 0,291 |

**Befund 1 — `window=24` ist kein Suchbudget-Artefakt.** Auch im
erschöpfenden Grid wählt die HPO in **8 von 9 Jobs `window=24`** (einzige
Ausnahme `raw/wmz_3`, ein degenerierter Job mit Test-F1 ≈ 0). Für
`trend/wmz_1` rangiert `window=48` nur auf **Platz 5** (Val-AUC 0,512)
hinter `window=24` (0,522). Die ursprüngliche Vermutung „Random-Search-
Budget zu klein" (§ 12.2/§ 12.6) ist damit **nicht haltbar** — `window=24`
ist die genuine Wahl des Selektionskriteriums, nicht ein Stichproben-Loch.

**Befund 2 — das eigentliche Problem ist ein Kriteriums-Mismatch.** Die
Val-ROC-AUC-Werte über die Fenstergrößen liegen extrem dicht beieinander
(auf der Trend-Schiene alle bei ~0,50–0,52); das Kriterium **unterscheidet
die Fenster praktisch nicht** — die Auswahl ist quasi indifferent/
verrauscht. Gleichzeitig liefert `window=48` laut Sweep ein besseres
**Test-F1**. Nicht das Such*budget* war also zu klein, sondern das
**Selektions*kriterium* (Validierungs-ROC-AUC) ist zu flach**, um die
test-F1-optimale Fenstergröße zu identifizieren. Konsequenz für den
Ausblick: ein test-näheres bzw. F1-basiertes HPO-Kriterium (oder
Schwellenwert-bewusste Validierung) wäre der zielführende Hebel — nicht
mehr Such-Iterationen.

**Befund 3 — der LSTM-AE bleibt unter dem Klassik-Champion.** Auch mit den
gridwh-HPs schlägt der LSTM-AE in **keinem** der 9 Jobs den Klassik-
Champion; am nächsten `residual/wmz_1` (F1 0,512 vs. LOF 0,837) und
`trend/wmz_1` (0,620 vs. PELT 0,700). Die Gesamtaussage aus § 12.4 bleibt.

**Methodischer Hinweis / ursprünglicher Confound (L4):** Der oben diagnostizierte
gridwh-Lauf stammte vom **Mac (CPU)**, die Baseline (§ 12.2/12.3) vom
Trainings-Laptop (**CUDA**, Random-Search). Die Test-F1-Spalte oben war
daher zunächst **kein sauberes A/B** (Hardware und Suchstrategie vermischt).
*(Korrektur 2026-07-02: Die numerische Präzision war entgegen früherer
Formulierung **nie** Teil des Confounds — das LSTM rechnet auf allen Devices
durchgängig in `float32`, s. `src/models/lstm_ae.py` / Methodik § 7.4.)*
Dieser Confound wurde **am 2026-06-10 aufgelöst** (siehe Nachtrag).

#### 12.7.1 Nachtrag — gridwh auf der RTX 3080 Ti (CUDA): L4 aufgelöst

Der gridwh-Lauf wurde am **2026-06-10 auf der RTX 3080 Ti (CUDA, `float32`)**
wiederholt — gleiche Hardware und Präzision wie die Baseline (§ 12.3), womit der
Confound L4 entfällt. Selektion weiterhin nach Validierungs-ROC-AUC.

| Variante | WMZ | window | hidden | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| raw | wmz_1 | 24 | 64 | 0,013 | 0,708 | 0,298 |
| raw | wmz_2 | 24 | 32 | 0,005 | 0,505 | 0,152 |
| raw | wmz_3 | 24 | 64 | 0,000 | 0,345 | 0,170 |
| residual | wmz_1 | 24 | 32 | 0,014 | 0,721 | 0,316 |
| residual | wmz_2 | 24 | 16 | 0,005 | 0,564 | 0,172 |
| residual | wmz_3 | 24 | 16 | 0,006 | 0,387 | 0,180 |
| trend | wmz_1 | 24 | 16 | **0,548** | 0,361 | **0,622** |
| trend | wmz_2 | 72 | 16 | 0,281 | 0,317 | 0,652 |
| trend | wmz_3 | 24 | 64 | 0,000 | 0,434 | 0,691 |

**Befund (confound-frei):** Die Kernaussage **hält auch ohne L4** — der LSTM-AE
bleibt in **allen 9 Jobs** unter dem jeweiligen Klassik-Champion (bester Job
`trend/wmz_1` F1 0,548 vs. PELT 0,700). Damit ist „Klassik genügt" nun **nicht
mehr durch einen Hardware-Confound angreifbar**. Nebenbefund: `trend/wmz_2`
verbessert sich gegenüber der Baseline deutlich (F1 0,001 → 0,281), bleibt aber
unter PELT (0,531). Die GPU-gridwh-PR-AUC sind zugleich die einzigen
**schwellenfreien LSTM-PR-AUC** auf same-hardware-Basis (vgl. § 11.2 †).

**Reproduktion:** `python src/stage8_hpo.py --models lstm_ae --lstm-strategy gridwh
--device cuda` + `python src/stage10_evaluate.py --models lstm_ae --device cuda`
(RTX 3080 Ti). Artefakte: `best_hparams_gridwh_gpu_roc_auc.json`,
`stage10_metrics_gridwh_gpu_roc_auc.csv` (in `outputs/gebaeude_a/hpo/` bzw.
`reports/`; Schreib-Spiegel auch in `export_schreiben_2026-06-11/`).

### 12.8 Folge-Experiment Option E (durchgeführt) und Ausblick (F)

Aus § 12.7 ergab sich **Option E** — die PR-AUC-Auswahl als Test des
Kriteriums-Mismatch. Sie wurde **am 2026-06-10 auf der RTX 3080 Ti durchgeführt**
(zusammen mit dem Confound-Rerun § 12.7.1). Die globale/pooled-Variante (F) wird
**nicht umgesetzt** und ist als Ausblick dokumentiert (Entscheidung 2026-06-04,
Begründung unten).

**(E) PR-AUC-Kriterium + Seed-Averaging — durchgeführt 2026-06-10.**
gridwh erneut auf der RTX, aber Auswahl nach **PR-AUC** statt Val-ROC-AUC
(`--hpo-metric pr_auc --hpo-seeds 3`). PR-AUC fokussiert die seltene
Positiv-Klasse (Davis und Goadrich 2006; Saito und Rehmsmeier 2015); Seeds
dämpfen die Init-Varianz (Reimers und Gurevych 2017; Bouthillier u. a. 2021).
Frage: Wählt das schärfere Kriterium die test-stärkere Fenstergröße (Hypothese
`window=48` auf `trend/wmz_1`)?

| Variante | WMZ | window | hidden | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| raw | wmz_1 | 24 | 32 | 0,017 | 0,676 | 0,275 |
| raw | wmz_2 | 72 | 32 | 0,000 | 0,442 | 0,138 |
| raw | wmz_3 | 24 | 16 | 0,000 | 0,322 | 0,167 |
| residual | wmz_1 | 24 | 16 | 0,014 | 0,717 | 0,315 |
| residual | wmz_2 | 72 | 16 | 0,000 | 0,479 | 0,148 |
| residual | wmz_3 | 24 | 16 | 0,006 | 0,387 | 0,180 |
| trend | wmz_1 | 24 | 32 | **0,614** | 0,367 | **0,624** |
| trend | wmz_2 | 24 | 16 | 0,000 | 0,515 | 0,726 |
| trend | wmz_3 | 48 | 32 | 0,000 | 0,456 | 0,669 |

**Befund (E):** Das schärfere PR-AUC-Kriterium **rettet den LSTM-AE nicht**.
`trend/wmz_1` erreicht F1 = 0,614 — praktisch identisch zur Baseline (0,611) und
weiter unter PELT (0,700). Die HPO wählt weiterhin überwiegend `window=24`
(`window=48` nur bei `trend/wmz_3`, dort F1 ≈ 0). Damit ist die
**`window=48`-Hypothese aus dem Daten-Sweep (§ 13) endgültig widerlegt** — ein
test-näheres Kriterium ändert die Selektion nicht zugunsten der vom Sweep
angedeuteten Fenstergröße.
Artefakte: `best_hparams_option_e_pr_auc_gpu.json`,
`stage10_metrics_option_e_pr_auc_gpu.csv` (in `outputs/gebaeude_a/hpo/` bzw.
`reports/`; Schreib-Spiegel auch in `export_schreiben_2026-07-13.zip`).

#### 12.8.1 Nachtrag 2026-07-13 — Pfad B (Seed-Std) und Pfad C (F1-HPO) durchgeführt

Zwei ergänzende Läufe auf der RTX 3080 Ti Mobile schließen die zwei bis
dahin offenen Lücken: (i) die trivial-`std = 0` in `lstm_seed_variance.csv`
(Persistierung des Seed-Mittelwerts vor dem Stage-8-Patch) und (ii) die
noch fehlende **dritte Selektionsmetrik F1** als Ablation neben ROC-AUC
und PR-AUC. Kombi-Wrapper: `outputs/nightrun_paths_b_and_c.ps1`;
Gesamtdauer 3,54 h (Pfad B) + 3,60 h (Pfad C).

**(B) Seed-Std-Persistierung — Option E mit `n_seeds = 3` und
per-Seed-Logging.** Der Stage-8-Patch (2026-06-11) schreibt jetzt eine
Zeile pro Seed in `hpo_log.csv` (Spalte `seed`); die Selektion bleibt der
Seed-Mittelwert, aber die Inter-Seed-Std ist rekonstruierbar
(Reimers und Gurevych 2017; Bouthillier u. a. 2021). Ergebnis in
`lstm_seed_variance.csv` (Snapshot `option_e_pr_auc_gpu_seedstd`):

| Variante | WMZ | best_mean | best_std | gap | robust_2σ |
|---|---|---:|---:|---:|:---:|
| raw | wmz_1 | 0,1771 | 0,0012 | 0,0014 | ✗ |
| raw | wmz_2 | 0,1713 | 0,0197 | 0,0019 | ✗ |
| raw | wmz_3 | 0,1304 | 0,0033 | 0,0025 | ✗ |
| residual | wmz_1 | 0,2181 | 0,0058 | 0,0054 | ✗ |
| residual | wmz_2 | 0,1862 | 0,0102 | 0,0044 | ✗ |
| residual | wmz_3 | 0,1262 | 0,0011 | 0,0003 | ✗ |
| trend | wmz_1 | 0,8110 | 0,0008 | 0,0003 | ✗ |
| trend | wmz_2 | 0,7915 | 0,0140 | 0,0111 | ✗ |
| trend | wmz_3 | 0,6050 | 0,0298 | 0,0113 | ✗ |

**Kernaussage:** `robust_2σ = 0/9`. In **keinem** der neun Jobs liegt der
Abstand zwischen Best- und Runner-up-HP oberhalb der 2σ-Seed-Unsicherheit
der Best-HP (Reimers und Gurevych 2017; Bouthillier u. a. 2021).
Die LSTM-AE-HP-Wahl ist damit über alle Selektions-Kriterien hinweg
**statistisch beliebig** — jede Wiederholung mit einem anderen Seed-Tripel
hätte in jedem Job eine andere `(window, hidden)`-Kombination gewählt.
L6 in § 15 ist damit **belegt statt offen**.

**(C) F1-HPO als dritte Selektionsmetrik — Snapshot `option_e_f1_gpu`.**
Selber Grid, aber `--hpo-metric f1` (point-adjusted F1 an der
Val-Quantil-Schwelle, Lipton u. a. 2014). Der Vergleich der drei
Selektionskriterien (ROC-AUC / PR-AUC / F1) je Job:

| Variante | WMZ | ROC-AUC | PR-AUC (Option E) | F1 (2026-07-13) | Test-F1 (F1-HPO) | Test-AUC (F1-HPO) |
|---|---|---:|---:|---:|---:|---:|
| raw | wmz_1 | (24, 64) | (24, 32) | (24, 16) | 0,014 | 0,670 |
| raw | wmz_2 | (24, 32) | (72, 32) | (72, 16) | 0,000 | 0,438 |
| raw | wmz_3 | (24, 64) | (24, 16) | (24, 32) | 0,006 | 0,316 |
| residual | wmz_1 | (24, 32) | (24, 16) | (72, 64) | 0,016 | 0,557 |
| residual | wmz_2 | (24, 16) | (72, 16) | (72, 32) | 0,000 | 0,469 |
| residual | wmz_3 | (24, 16) | (24, 16) | (24, 16) | 0,006 | 0,387 |
| trend | wmz_1 | (24, 16) | (24, 32) | (24, 16) | **0,548** | 0,361 |
| trend | wmz_2 | (72, 16) | (24, 16) | (24, 64) | 0,211 | 0,527 |
| trend | wmz_3 | (24, 64) | (48, 32) | (72, 64) | 0,000 | 0,467 |

**Zentrale Befunde (C):**

- **Die drei Selektoren wählen in 8 von 9 Jobs verschiedene
  `(window, hidden)`-Kombinationen** (Übereinstimmung nur bei
  `residual/wmz_3`). In Kombination mit Pfad-B-Befund `robust_2σ = 0/9`
  ist das nachträglich erwartbar — die HP-Landschaft ist zu flach, jedes
  Kriterium landet in einem anderen Rausch-lokalen Optimum.
- **F1 als Kriterium ist strukturell spröde:** bei `residual/wmz_1`
  liefern 8 von 9 Val-Configs F1 = 0,0000 (nur eine 0,061). Grund: F1
  fällt hart auf 0, sobald das Val-Quantil das Score-Ranking knapp
  verfehlt (0 Precision oder 0 Recall). PR-AUC ist als glattes
  Ranking-Kriterium **strukturell überlegen** (Davis und Goadrich 2006;
  Saito und Rehmsmeier 2015).
- **F1-HPO liefert keinen Test-F1-Durchbruch.** `trend/wmz_1` fällt
  gegenüber PR-AUC-Wahl **von 0,614 auf 0,548** — direkter Beleg, dass
  ein test-nahes Kriterium bei zu flacher Val-Landschaft *schlechter*
  wählen kann als ein glattes. Einzige Verbesserung: `trend/wmz_2`
  0,000 → 0,211 (aber Test-AUC bleibt bei 0,527, effektiv Zufall).
- **Score-Inversion bleibt strukturell:** die neue
  `score_distribution_diagnose.csv` (mit F1-Champion) klassifiziert das
  LSTM-AE-Muster als `inversion` auf `raw/wmz_3` und **allen drei
  trend-WMZ**, sonst als `overlap`. Klassik-Champions haben **nie**
  Inversion (IF: overlap oder threshold-shift; LOF: overlap; PELT:
  overlap). Damit reproduziert eine dritte unabhängige Selektionsmetrik
  die zentrale Diagnose.

**Gesamt-Konsequenz (E + B + C zusammen):** Die LSTM-AE-Unterlegenheit
ist unter drei Selektionskriterien × drei Seed-Wiederholungen ×
strukturell-diagnostischer Score-Verteilungs-Analyse robust reproduziert.
Der Negativbefund gegen H2 ist damit **maximal abgesichert** —
methodische Angriffsflächen „HPO zu klein / Selektor falsch / Seed-
Rauschen / Rekonstruktion kaputt" sind alle geprüft und keine erklärt
die Lücke zur Klassik.
Artefakte: `best_hparams_option_e_pr_auc_gpu_seedstd.json`,
`best_hparams_option_e_f1_gpu.json`, `hpo_log.csv` (mit `seed`-Spalte,
243 LSTM-Zeilen pro Selektor-Lauf), `lstm_seed_variance.csv`,
`score_distribution_diagnose.csv`, `score_verteilungen.png` — Schreib-
Spiegel im Export `export_schreiben_2026-07-13.zip`.

#### 12.8.2 Nachtrag 2026-07-20 — Score-Ablation durchgeführt (RTX)

Aus § 12.4/12.7 und der Rekonstruktions-Diagnostik (Methodik § 7.12)
folgte die Hypothese, dass die LSTM-AE-Schwäche im Score-Mapping und
nicht in der Repräsentation sitzt — der `window_mse`-Baseline mittelt
einen punktscharfen Rekonstruktionsfehler über bis zu 24 Fenster hinweg
auf ≈ 0 (Malhotra u. a. 2016). Zur kausalen Prüfung wurde jedes trainierte
LSTM-AE-Modell (Stage-8-Best-HPs pro Job) unter vier Scorings am
identischen Bewertungs-Maßstab (Stage-10-Protokoll: Val-0,99-Quantil je
Modus, PA-F1, gleiches injiziertes Test-Set) neu bewertet:
`window_mse` (Baseline, MSE über Fenster × alle Features), `channel_mse`
(nur Ziel-Kanal), `last_step_mse` (nur letzter Zeitschritt = punktschärfste
Attribution), `mahalanobis` (Originalscoring EncDec-AD; Malhotra u. a.
2016). Runner: `src/tools/lstm_ae_score_ablation.py --device cuda`,
Wandzeit 8:08 min GPU (Re-Fit dominiert). Ergebnis in
`reports/lstm_ae_score_ablation.csv` (36 Zeilen = 9 Jobs × 4 Modi).

**Ergebnisse (point-adjusted F1 je Modus, alle 9 Jobs):**

| Variante | WMZ | window_mse | channel_mse | last_step_mse | mahalanobis | Δ_max | Klassik-Champ |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | wmz_1 | 0,014 | 0,015 | **0,548** | 0,372 | **+0,534** | 0,911 (IF) |
| raw | wmz_2 | 0,000 | 0,000 | 0,008 | 0,005 | +0,008 | 0,292 (LOF) |
| raw | wmz_3 | 0,006 | 0,018 | 0,006 | 0,006 | +0,012 | 0,006 (Z) |
| residual | wmz_1 | 0,016 | 0,444 | **0,705** | 0,034 | **+0,689** | 0,837 (LOF) |
| residual | wmz_2 | 0,000 | 0,000 | **0,064** | 0,060 | **+0,064** | 0,322 (LOF) |
| residual | wmz_3 | 0,006 | 0,006 | 0,019 | 0,019 | +0,013 | 0,019 (Z) |
| trend | wmz_1 | 0,548 | 0,548 | 0,554 | 0,554 | +0,006 | 0,700 (PELT) |
| trend | wmz_2 | 0,211 | 0,211 | 0,001 | 0,001 | +0,000 | 0,531 (PELT) |
| trend | wmz_3 | 0,000 | 0,000 | 0,000 | 0,000 | +0,000 | 0,000 (PELT) |

**Zentrale Befunde:**

- **Score-Reparatur in 3 von 9 Jobs empirisch belegt** (Δ F1 ≥ 0,05):
  raw/wmz_1 (+0,534), residual/wmz_1 (+0,689), residual/wmz_2 (+0,064).
  Der wirkungsvollste Modus ist konsistent **`last_step_mse`** — die
  reine Fenstermittelung (`window_mse`) verwässert punktscharfe Attributionen
  über das Rekonstruktionsfenster hinweg.
- **6 von 9 Jobs bleiben schwach** (max F1 ≤ 0,211 auch unter dem besten
  alternativen Scoring): raw/wmz_{2,3}, residual/wmz_3, alle drei trend-Jobs.
  Dort ist der Score-Mode **nicht** der dominante Erklärer der Unterlegenheit.
- **Kein Klassik-Schlag unter irgendeinem Scoring:** Der beste je erreichte
  LSTM-AE-Wert (residual/wmz_1 = 0,705 unter last_step_mse) unterliegt
  weiter dem LOF-Champion (0,837); auf raw/wmz_1 sind es 0,548 vs. 0,911
  (IF). In keinem Job mit substanzieller Klassik-Referenz (F1 > 0,05)
  schließt ein alternatives LSTM-AE-Scoring die Lücke.
- **Regressionen möglich:** trend/wmz_2 bricht unter `last_step_mse` bzw.
  `mahalanobis` von 0,211 auf 0,001 ein — alternative Scorings sind
  **nicht monoton** überlegen, ihre Wahl bleibt bewertungs- statt
  konstruktions-getrieben.
- **Score-Inversion persistiert:** In sechs Jobs bleibt in mindestens
  einem Modus ROC-AUC < 0,5; auf trend/wmz_1 liegt sie in **allen vier**
  Modi zwischen 0,361 und 0,393. Der Modus ändert also nicht die
  Rang-Inversion — das Problem sitzt in der gelernten Distanzstruktur, nicht
  nur in ihrer Reduktion auf einen Skalar.

**Interpretation für Kap. 4/5.** Die Score-Ablation liefert einen
*differenzierten*, kausal fundierten Befund: (i) In den beiden wmz_1-Jobs
(raw und residual — genau die Jobs mit klaren Punktanomalien in der
Anomalie-Injektion) ist `window_mse` messbar suboptimal; ein
punktschärferes Scoring reduziert die Lücke zum Klassik-Champion um mehr
als eine Größenordnung. (ii) In den übrigen sieben Jobs ist die
Score-Mapping-Reparatur *nicht* das dominante Problem — dort ist die
gelernte Repräsentation zu unspezifisch oder die Anomalie-Signatur passt
nicht zur Rekonstruktions-Perspektive. (iii) Damit ist die
LSTM-AE-Unterlegenheit **nicht** monokausal auf „falsches Scoring"
zurückführbar; sie ist bestenfalls in einem Drittel der Jobs
score-basiert reparabel, verbleibt aber selbst dort strukturell unterhalb
der Klassik. In der Fünf-Ebenen-Diskussion (Kap. 5.6) ersetzt dieser
Befund die bisher qualitative Ebene 4 („Score-Mapping bricht") durch
eine quantifizierte Formulierung: *„Score-Mapping ist Mit-Ursache in
3/9 Jobs; in 7/9 nicht das dominante Problem — die Rekonstruktions-
Erklärung genügt nicht."* Artefakte: `reports/lstm_ae_score_ablation.csv`
(alle Metriken je Modus), Lauf-Log `outputs/_logs/score_ablation_*.log`.

**(F) Globales/pooled Training über alle WMZ — Ausblick, nicht umgesetzt.**
Bislang trainiert die Pipeline **pro WMZ getrennt** (lokale Modelle). Ein
einziges LSTM-AE über die Fenster **aller drei Zähler** (geteilte Gewichte)
ist das einzige Setup, in dem ein KI-Modell einen *strukturellen* Vorteil
gegenüber der Klassik hat — Z-Score/LOF/IF/PELT lernen nicht über Zähler
hinweg. Erwarteter Nutzen: 3× Trainingsdaten (stabilerer Fit, geringere
Seed-Varianz), geteilter Prior (Transfer auf den schwachen wmz_1), implizite
Regularisierung, Skalierbarkeit (Montero-Manso und Hyndman 2021; mit nur 3
Serien Effekt begrenzt, negativer Transfer möglich). **Warum nicht
umgesetzt:** Der Korrelationsbefund (unten) begrenzt den Mehrwert auf das
Residuum und auf einen mit nur drei Zählern marginalen Effekt; der Aufwand
(Pooling-Refactor) steht dazu in keinem Verhältnis. Als *Ausblick* bleibt
festgehalten, dass ein globales Modell der einzige Kandidat mit einem
strukturellen KI-Vorteil wäre — relevant erst bei vielen/heterogenen Zählern
(Kap. 6). Design/Refactor als Referenz in HANDOFF.md § 9.2, Methodik § 7.11.

**Voraussetzungs-Check (Korrelationsanalyse, bereits durchgeführt).** Vor
der Implementierung von (F) wurde geprüft, ob Pooling überhaupt
gerechtfertigt ist: `src/exploration/cross_meter_correlation.py` misst die
paarweise WMZ-Korrelation je Signalebene.

| Paar | Roh-kW (A) | MSTL-Trend | **Residuum (B)** |
|---|---:|---:|---:|
| wmz_1 ↔ wmz_2 | 0,41 | 0,61 | **0,06** |
| wmz_1 ↔ wmz_3 | 0,47 | 0,60 | **0,03** |
| wmz_2 ↔ wmz_3 | 0,74 | 0,83 | **0,35** |

Die Korrelation steckt fast ausschließlich im **Trend** (gemeinsamer
Saison-/Temperaturtreiber); das **anomalie-relevante Residuum ist nahezu
unkorreliert** (wmz_1 gegen beide ≈ 0). Zusätzlich liegen die Roh-Skalen
weit auseinander (wmz_3 ≈ 95 kW Mittel vs. wmz_1 ≈ 10 kW). Daraus folgt:

- **Pooling (F) ist nur auf Variante B (Residuum) gerechtfertigt** — niedrige
  Redundanz (≈ 0 → echtes 3×-Daten) bei durch STL/Normalisierung
  homogenisierter Skala/Saison. Auf raw (A) ungünstig (Skalen-Heterogenität
  → negativer Transfer, v. a. wmz_1; höhere Redundanz).
- **Das multivariate Design (b) / Anomaly Transformer ist von den Daten
  nicht gestützt:** auf der Residuum-Ebene gibt es keine zähler-
  übergreifende Korrelation auszunutzen → bleibt im Ausblick.
- Nebenbefund: Die ≈-null Residuum-Korrelation **bestätigt nachträglich die
  Per-Zähler-Architektur** — die Abweichungen vom Normalmuster laufen je
  Zähler unabhängig.

**Diagnostische Verzahnung & Objektivität.** Die drei Achsen werden am
**eingefrorenen Bewertungs-Maßstab** (§ 7.7: dasselbe Test-Set, dieselbe
deterministische Injektion, dieselben Metriken) gegeneinander gehalten —
nur Kandidat bzw. Trainings-Input variiert, nie das Lineal:
- **Datenmenge** → Daten-Sweep (§ 7.8, § 13/§ 12.6): Ist das LSTM
  datenlimitiert? Er ist der **Vorab-Test für (F)** (nur wenn die Lernkurve
  noch steigt, verspricht Pooling Gewinn) und lieferte den Zielwert
  (window=48 → F1 0,776), den **(E)** mit einem fairen Kriterium *ohne*
  Test-Blick treffen sollte — **(E) ist durchgeführt (§ 12.8): das Kriterium
  wählt weiter `window=24`, der Zielwert wird nicht getroffen.** Wichtig: Der
  Sweep fügt Daten *desselben* Zählers
  hinzu, Pooling Daten *anderer* — (F) lohnt nur, wenn **Sweep** (Kurve
  steigt) **und** **Korrelationsanalyse** (Zähler ähnlich/nicht redundant)
  beide zutreffen.
- **Kriterium** → Option E. **Lokal vs. global** → Option F. Beide nutzen
  denselben fixen Maßstab → objektiv und attribuierbar.

> **Hinweis zur Heterogenität:** Die raw/residual-Varianten sind durch die
> Außentemperatur bereits *multivariat*; echte Quellen-*Heterogenität*
> (mehrere Gebäude/Energieträger/Qualitätsstufen) entsteht erst mit dem
> zweiten Datensatz (Stage 11, § 7.9). Das KI-Potenzial wird dort am
> ehesten sichtbar.

---

## 13 Daten-Sweep — Datenmengen-Sensitivität (Klassik + LSTM-AE)

Neues Diagnose-Skript: [src/data_sweep.py](../src/tools/data_sweep.py).
Es trainiert **einen wählbaren Detektor** (Klassik oder LSTM-AE) mit
identischen Hyperparametern auf verschieden großen Subsets des
Trainings-Tails (zusammenhängend, jüngster Block) und evaluiert auf
**derselben** Validierung und demselben injizierten Test-Set wie
Stage 10. Damit sind die Stufen direkt vergleichbar — Val und Test
sind in Stage 5 zeitlich fixiert und werden nicht angefasst. Per
Default werden die HPs aus `best_hparams.json` geladen (= dieselben
wie in Stage 10), sodass die Sweep-Stufe `train_rows=0` direkt mit
dem Stage-10-Wert vergleichbar ist.

Beispielaufruf:

```powershell
python src\data_sweep.py --variant trend --wmz wmz_1 `
    --train-rows 2000 5000 10000 0 `
    --window-size 48 --hidden-size 32 --n-layers 2 --epochs 30
```
(0 in der Liste = „alle verfügbaren Flag-bereinigten Trainings-Stunden".)

**Beobachtete Stufen (auf RTX 3080 Ti Mobile, Default-HPs):**

*raw/wmz_1 (window=24, hidden=32, layers=2, epochs=30):*

| Train-Stunden | fit-Dauer | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 2 000 (8,3 % von 23 950) | 4,6 s | 0,024 | 0,010 | 0,014 | **0,726** |
| 5 000 (20,9 %) | 6,7 s | 0,024 | 0,010 | 0,014 | 0,717 |
| 10 000 (41,8 %) | 13,9 s | 0,021 | 0,010 | 0,013 | 0,717 |
| 23 950 (100 %) | 28,0 s | 0,019 | 0,010 | 0,013 | 0,649 |

*trend/wmz_1 (window=48, hidden=32, layers=2, epochs=30):*

| Train-Stunden | fit-Dauer | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 2 000 (7,7 % von 25 872) | 4,8 s | 0,739 | **0,808** | 0,772 | 0,324 |
| 5 000 (19,3 %) | 6,9 s | 0,729 | 0,808 | 0,766 | 0,310 |
| 10 000 (38,7 %) | 14,5 s | **0,746** | **0,808** | **0,776** | 0,284 |
| 25 872 (100 %) | 33,7 s | 0,715 | 0,534 | 0,611 | 0,319 |

*Nachtrag mit HPs aus dem vollen LSTM-HPO (window=24, hidden=32, layers=2, lr=0,0003, epochs=100):*

| Train-Stunden | fit-Dauer | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 2 000 | 11,9 s | 0,739 | 0,808 | **0,772** | 0,341 |
| 5 000 | 23,5 s | 0,731 | 0,808 | 0,768 | 0,333 |
| 10 000 | 46,5 s | 0,692 | 0,535 | 0,603 | 0,321 |
| 25 872 (Stage-10-Wert) | 117,2 s | 0,712 | 0,535 | 0,611 | 0,288 |

> **Welche Zahlen stehen in der Arbeit?** MA Tabelle 6 führt die Zeile
> `lstm_ae / trend` seit dem 2026-07-28 mit den **Nachtrags-Werten**
> (`window=24`, 0,772 / 0,768 / 0,603 / 0,611). Gründe: Diese HPs sind die
> vom HPO gewählten Champion-HPs (Tabelle 7), damit verwenden alle vier
> Tabellenzeilen denselben HP-Satz; die Werte stehen so in
> [data_sweep.csv](../outputs/gebaeude_a/reports/data_sweep.csv) und sind
> direkt reproduzierbar; und **Abb. 14 (Lernkurven) speist sich aus derselben
> CSV** — mit den `window=48`-Werten widersprächen sich Tabelle und Abbildung
> an der Stufe 10 000 (Maximum 0,776 vs. Einbruch 0,603). Die
> `window=48`-Zeile bleibt hier als historischer Lauf dokumentiert; die daraus
> abgeleitete Hypothese ist ohnehin widerlegt (§ 12.7 / § 12.8).

**Schlüssel-Befund:** Mit den frisch gefundenen besten HPs ist `F1` bei
2 000 Trainings-Stunden = **0,772** — und sinkt monoton zur vollen Menge
hin auf 0,611. Das bestätigt die Hypothese aus § 14 Punkt 8 / § 12.6
Punkt 1: die HPs sind für kleine Trainingsmengen optimiert (vermutlich
weil ein Großteil der Validierungs-Anomalien punktuelle Spikes sind, die
schon mit wenig Kontext rekonstruierbar werden). Ein in der Thesis
diskutables Ensemble-Setting wäre: **LSTM-AE auf 2 000–5 000 jüngsten
Stunden + PELT auf vollem Trend** — die beiden Detektoren würden sich
auf der Trend-Schiene komplementär ergänzen.

*Weitere LSTM-AE Champion-Sweeps mit vollen HPs:*

| Job | Train-Stunden | F1 | Bemerkung |
|---|---:|---:|---|
| raw/wmz_1/lstm_ae (window=24,hidden=32,layers=2,lr=0,01,epochs=50) | 2 000 | **0,765** | Bestwert! |
| raw/wmz_1/lstm_ae | 5 000 | 0,014 | komplett zusammengebrochen — instabil |
| raw/wmz_1/lstm_ae | 10 000 | 0,394 | erholt sich teilweise |
| raw/wmz_1/lstm_ae | 23 950 (Stage-10) | 0,212 | erneut Overfit |
| residual/wmz_1/lstm_ae (window=24,hidden=64,layers=2,lr=0,001,epochs=100) | 2 000 | 0,228 | Bestwert auf dieser Schiene |
| residual/wmz_1/lstm_ae | 5 000 / 10 000 / voll | 0,014 / 0,013 / 0,014 | gesättigt auf niedrigem Niveau |

**Drei zusätzliche Befunde aus den vollen LSTM-Sweeps:**

5. **Instabilität bei mittleren Trainingsmengen.** `raw/wmz_1` zeigt
   ein extremes Muster: F1 = 0,765 bei 2 000 → 0,014 bei 5 000 → 0,394
   bei 10 000. Das spricht für eine sehr empfindliche Loss-Landschaft
   bei dieser HP-Konfiguration — vermutlich ist `learning_rate=0,01`
   für 5 000 Stunden zu hoch und führt zu schlechter Konvergenz.
6. **Frühe Sättigung bei residual/wmz_1.** Schon ab 5 000 Stunden hat
   F1 sein Plateau auf 0,014 erreicht, das sich auch bei voller Menge
   nicht ändert. Hier limitiert eindeutig die HP-Konfiguration, nicht
   die Datenmenge.
7. **Konsistent niedrige bis negative AUC auf trend-Schiene** (alle
   Stufen: 0,288–0,341). Bestätigt die in § 11.2 diskutierte Inversions-
   Hypothese: der LSTM-AE rekonstruiert die Anomalie-Stunden
   *besser* als die normalen Übergänge, weshalb die Scores systematisch
   verkehrt herum rangieren.

*raw/wmz_1/iforest (HPs aus best_hparams.json: n_estimators=400, max_features=1,0):*

| Train-Stunden | fit-Dauer | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 2 000 (8,3 %) | 0,9 s | 0,929 | 0,314 | 0,469 | 0,733 |
| 5 000 (20,9 %) | 0,3 s | 0,774 | 0,104 | 0,183 | 0,766 |
| 10 000 (41,8 %) | 0,3 s | 0,891 | 0,862 | 0,876 | 0,812 |
| 23 950 (100 %) | 0,5 s | **0,876** | **0,949** | **0,911** | **0,836** |

(Vollwert reproduziert exakt die Stage-10-Zeile aus § 11.2.)

**Drei nicht-triviale Befunde:**

1. **Mehr Daten ist nicht immer besser.** Auf beiden Varianten sinkt die
   Performance bei voller Trainingsmenge gegenüber 10 000-Stunden-Subsets.
   Das ist klassisches Overfitting bei fixen HPs (`window`, `hidden`,
   `epochs`). Das volle HPO muss daher beides gemeinsam tunen — die
   Hyperparameter und die effektive Trainingsmenge stehen in
   Wechselwirkung.
2. **trend/wmz_1 schlägt PELT bereits ohne HPO.** Der Sweep-Bestwert
   F1 = 0,776 (bei 10 000 Trainings-Stunden) liegt **deutlich über** dem
   PELT-Optimum F1 = 0,700 aus Stage 10. Damit gibt es schon vor dem
   nächtlichen HPO einen ersten harten Beleg, dass der LSTM-AE auf der
   Trend-Schiene komplementär zum PELT-Verfahren wirken kann.
3. **Fit-Dauer skaliert annähernd linear mit Trainings-Stunden** (Faktor
   ~6 für 12× mehr Daten, etwas besser als linear wegen Setup-Overhead).
   Hochrechnung: voller HPO-Lauf mit 25 Configs × 9 Jobs × ~30 s/fit ≈
   115 min im optimistischen Fall, eher 5–10 h wenn die HPO große
   `window`/`hidden`/`epochs`-Kombinationen zieht.
4. **IsolationForest zeigt eigenes Sättigungs­muster:** bei 5 000
   Trainings-Stunden bricht F1 auf 0,183 ein (zu wenig Baumvielfalt),
   stabilisiert sich aber ab 10 000 Stunden auf ~0,88. Damit sind
   **~10 000 Stunden ein guter „Sättigungs"-Wert** für IForest auf
   `raw/wmz_1` — die letzten ~14 000 Stunden bringen nur noch +0,04
   F1. Konsequenz für die Thesis-Diskussion: pro Modell ist die
   Mindest-Trainingsmenge unterschiedlich, das ist nicht ein
   einzelner „magischer Schwellwert" für alle Detektoren.

### 13.1 Klassik-Champion-Sweeps (alle 9 WMZ × Variant-Kombinationen)

Vollständige Datenmengen-
Sensitivität für die drei Klassik-Champions
(IsolationForest auf raw, LOF auf residual, PELT auf trend), jeweils
mit den besten HPs aus [hpo/best_hparams.json](../outputs/gebaeude_a/hpo/best_hparams.json)
und Train-Stufen `2 000 → 5 000 → 10 000 → voll`.

*raw / IsolationForest:*

| WMZ | 2 000 | 5 000 | 10 000 | voll | Stage-10-Vergleich |
|---|---:|---:|---:|---:|---:|
| wmz_1 | F1=0,469 (AUC=0,733) | F1=0,183 | F1=0,876 | **F1=0,911** | F1=0,911 ✓ |
| wmz_2 | F1=0,003 | F1=0,000 | F1=0,005 | F1=0,005 | F1=0,005 ✓ |
| wmz_3 | F1=0,000 | F1=0,006 | F1=0,006 | F1=0,006 | F1=0,006 ✓ |

*residual / LOF:*

| WMZ | 2 000 | 5 000 | 10 000 | voll | Stage-10-Vergleich |
|---|---:|---:|---:|---:|---:|
| wmz_1 | F1=0,749 | **F1=0,895** | F1=0,844 | F1=0,837 | F1=0,837 ✓ |
| wmz_2 | F1=0,055 | F1=0,010 | F1=0,431 | F1=0,322 | F1=0,322 ✓ |
| wmz_3 | F1=0,019 | F1=0,019 | F1=0,019 | F1=0,006 | F1=0,006 ✓ |

*trend / PELT:*

| WMZ | 2 000 | 5 000 | 10 000 | voll | Stage-10-Vergleich |
|---|---:|---:|---:|---:|---:|
| wmz_1 | F1=0,700 | F1=0,700 | F1=0,700 | F1=0,700 | F1=0,700 ✓ |
| wmz_2 | F1=0,531 | F1=0,531 | F1=0,531 | F1=0,531 | F1=0,531 ✓ |
| wmz_3 | F1=0,000 | F1=0,000 | F1=0,000 | F1=0,000 | F1=0,000 ✓ |

(Die Stage-10-Vergleichsspalte zeigt, dass die volle Trainingsmenge
exakt die Zahlen aus § 11.2 reproduziert — der Sweep-Mechanismus ist
damit konsistent verifiziert.)

**Modellspezifische Sättigungs- und Sensitivitäts-Muster:**

1. **PELT ist invariant gegenüber der Trainingsmenge.** Alle vier
   Stufen liefern identische F1/Precision/Recall/AUC. Erklärung:
   PELT wird in Stage 10 (und im Sweep) auf der **vollen
   Trend-Reihe** angewendet — die Train-Subset-Größe steuert nur die
   Auswahl der Validierungs-Schwelle, und die ist bei PELT ohnehin
   binär (0,5). Das ist konzeptionell konsistent mit der Methodik:
   PELT ist ein offline-Change-Point-Verfahren, kein parametrisches
   Modell mit Trainings-Dynamik. Methodische Folgerung für die
   Thesis: der Daten-Sweep ist für PELT nicht aussagekräftig, kann
   im Diskussions-Kapitel als Negativbefund („PELT hat keinen
   Datenhunger") angeführt werden.
2. **IsolationForest hat ein nicht-monotones Profil bei wmz_1**:
   2 000 → 5 000 sinkt F1 von 0,469 auf 0,183 (zu wenig Baum-
   Vielfalt für stabile Isolation), springt dann bei 10 000 auf
   0,876 und sättigt. **~10 000 Stunden sind der „Knick"** —
   darunter ist IForest instabil, darüber stabil.
3. **LOF hat ein klares Optimum unter dem Maximum**: residual/wmz_1
   erreicht bei 5 000 Stunden F1 = 0,895 (Bestwert), bei voller
   Trainingsmenge fällt es auf 0,837. Mehr Daten machen das
   Dichte-Modell „global" und schwächen die lokale Sensitivität.
   Empfehlung für künftige Erweiterungen: für LOF könnte ein
   gleitendes Trainings-Fenster (z. B. letzte 6 Monate) bessere
   Ergebnisse liefern als das volle 3-Jahres-Fenster.
4. **wmz_3 ist unrettbar:** auf allen Sweep-Stufen liefert kein
   Klassik-Modell brauchbare F1-Werte. Bestätigt die in § 11.3 (a)
   diskutierte Diagnose: 52 % geflaggte Stunden + nur 307 eligible
   Test-Stunden + viele echte Strukturbrüche aus den Sensor-
   Lücken — der Trainings­filter entfernt zu viele Stunden, und der
   Test selbst ist zu klein für signifikante Aussagen.

**Sweep-Output:** [data_sweep.csv](../outputs/gebaeude_a/reports/data_sweep.csv)
(append-only; weitere Sweep-Aufrufe für andere Variante/WMZ-Kombinationen
werden angehängt).

### 13.2 LSTM-AE-Sweep-Generalisierung auf wmz_2/wmz_3 (Nachtrag 2026-07-20)

Der bis dahin nur auf wmz_1 (raw/residual/trend) durchgeführte
LSTM-AE-Sweep wurde am 2026-07-20 auf `wmz_2` und `wmz_3` in den beiden
Schienen mit signifikantem Klassik-Signal (`raw` und `trend`) ausgeweitet
(die vier Läufe via `python src/tools/data_sweep.py --model lstm_ae
--variant {raw,trend} --wmz {wmz_2,wmz_3}` gestartet, Wandzeit 6:11 min
GPU). Ziel: Prüfung,
ob die auf wmz_1 gefundene **Datenmengen-Sensitivität** des LSTM-AE ein
zähler-lokales Artefakt oder ein systematisches Muster ist. Die
Klassik-Seite dieser vier Jobs liegt bereits seit dem Sweep vom
2026-06-01 in `data_sweep.csv` (§ 13.1).

*LSTM-AE F1 je Stufe (Stage-8-Best-HPs pro Job, alle vier neuen Jobs):*

| Variante | WMZ | HPs (window/hidden/lr/epochs) | 2 000 | 5 000 | 10 000 | voll | Klassik-Champion (voll) |
|---|---|---|---:|---:|---:|---:|---:|
| raw | wmz_2 | 72 / 16 / 0,001 / 50 | 0,000 | 0,000 | 0,000 | 0,000 | 0,005 (IF) |
| raw | wmz_3 | 24 / 32 / 0,001 / 50 | 0,000 | 0,000 | 0,000 | 0,006 | 0,006 (IF) |
| trend | wmz_2 | 24 / 64 / 0,001 / 50 | 0,000 | 0,000 | 0,000 | **0,211** | 0,531 (PELT) |
| trend | wmz_3 | 72 / 64 / 0,001 / 50 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 (PELT) |

*ROC-AUC je Stufe (Inversions-Check < 0,5):*

| Variante | WMZ | 2 000 | 5 000 | 10 000 | voll |
|---|---|---:|---:|---:|---:|
| raw | wmz_2 | 0,445 | 0,422 | 0,420 | 0,438 |
| raw | wmz_3 | 0,389 | 0,326 | 0,302 | 0,316 |
| trend | wmz_2 | 0,579 | 0,420 | 0,412 | 0,527 |
| trend | wmz_3 | 0,332 | 0,467 | 0,462 | 0,467 |

**Kernbefunde:**

- **Datenmenge ist nicht die Ursache der LSTM-AE-Schwäche auf wmz_2/wmz_3.**
  In allen vier neuen Jobs bleibt F1 an drei von vier Trainingsstufen bei
  exakt 0,000; nur `trend/wmz_2` zeigt bei voller Menge einen sprunghaften
  Übergang von 0,000 auf 0,211. Die Lernkurve ist damit über weite
  Bereiche flach — kein sichtbarer Grenznutzen zusätzlicher Trainingsdaten.
- **Kein Klassik-Schlag in irgendeinem der vier Jobs.** Der Vollwert liegt
  auf `raw/wmz_2`/`wmz_3` unterhalb oder gleich dem IsolationForest-
  Champion, auf `trend/wmz_2` bei 0,211 gegen PELT 0,531 (Faktor 2,5 zu
  wenig), auf `trend/wmz_3` unentschieden bei 0,000 (dort ist auch die
  Klassik bereits am Limit — L1/L3-Beleg).
- **ROC-AUC-Inversion persistiert** über alle Sweep-Stufen auf
  `raw/wmz_3` (0,302–0,389) und `trend/wmz_3` (0,332–0,467); auf
  `raw/wmz_2` grenzwertig (0,420–0,445), auf `trend/wmz_2` schwach positiv
  (0,412–0,579). Das Muster ist konsistent mit § 12.4/§ 12.8.2:
  Score-Inversion ist strukturell, nicht datenmengen-getrieben.
- **`trend/wmz_2`-Sprungpunkt (0 → 0,211 bei voller Menge).** Interpretation:
  Erst mit den vollen 25 872 Train-Stunden lernt der LSTM-AE das saisonale
  Grundprofil so gut, dass ein Teil der nicht-stationären Anomalien
  überhaupt einen von null verschiedenen Rekonstruktions-Score bekommt;
  darunter fehlt schlicht die saisonale Deckung. Selbst dieser Sprung
  reicht aber nicht an PELT (0,531) heran.

**Konsequenz für Kap. 5 (H2 und L1).** Die Erweiterung schließt die
zähler-lokale Interpretation aus: Die auf wmz_1 gemachten Beobachtungen
(Overfitting bei voller Menge, HP-Datenmenge-Wechselwirkung — § 13
oben) sind nicht der Alleinerklärer der Klassik-Überlegenheit; auf
wmz_2/wmz_3 bricht die LSTM-AE-Lernkurve bereits **vor** der
HP-Interaktion zusammen. Zusammen mit § 12.8.2 (Score-Ablation:
`window_mse`-Verwässerung ist Mit-Ursache in 3/9 Jobs, aber nicht
dominant) und § 12.8.1 (Seed-Robustheit als beliebige HP-Wahl belegt)
ist damit die dritte Angriffsfläche „zu wenig Daten" empirisch
zurückgewiesen — die H2-Bestätigung steht unter allen drei denkbaren
methodischen Confoundern (HP-Selektor, Seed-Rauschen, Datenmenge).
Artefakte: `reports/data_sweep.csv` (vollständig, 64 Zeilen inkl. der
neuen 16), Lauf-Log `outputs/_logs/data_sweep_*.log`.

## 14 Diskussion und Empfehlung für die Thesis

1. **Auf der raw-Schiene ist IsolationForest klar der Champion** für die
   sensor-saubere Anlage (`wmz_1`): F1 = 0,911 mit konkretem Recall pro
   Typ als Headline-Result.
2. **Auf der residual-Schiene gewinnt LOF.** Lesart: die residuale
   Repräsentation reduziert das Feature-Set auf wenige korrelierte
   Dimensionen, in denen Dichteschätzung tragfähiger ist als
   Baum-Isolation.
3. **PELT auf der Trend-Schiene erkennt Drift/Strukturbruch zuverlässig**
   und liefert mit der EnSikuMaV-Validierung einen labelfreien Echtwelt-
   Beleg. Dies ist methodisch der wertvollste Befund — er zeigt, dass der
   Detektor *nicht nur synthetische* Anomalien erkennt.
4. **Plateau-Anomalien bleiben die offene Flanke der klassischen Modelle.**
   Auch der LSTM-Autoencoder schließt die Lücke nicht (Plateau-Recall
   ≈ 0 auf allen Variante×WMZ-Kombinationen, s. § 12.3). Diskussions­
   würdig: das Plateau ist ein konstanter Signal-Abschnitt — weder
   Punkt-Ausreißer noch Drift, sondern „nichts" — und damit auch für
   einen Rekonstruktions-AE schwer detektierbar, weil eine konstante
   Sequenz nicht schlechter rekonstruiert wird als eine konstante
   Phase ohne Heizbedarf.
5. **LSTM-Autoencoder konnte in keinem der 9 Variante×WMZ-Setups den
   jeweiligen Klassik-Champion schlagen** (s. § 12.2 und Vergleichs­
   tabelle in § 12.4). Der nächstgelegene Wert: auf `trend/wmz_1`
   erreicht er F1 = 0,611 gegen PELT-F1 = 0,700. Der Daten-Sweep
   (§ 13 / § 12.5) hatte mit `window=48` (das HPO wählte `window=24`)
   schon ohne HPO bei 10 000 Stunden F1 = 0,776 angedeutet — woraus
   zunächst die Vermutung entstand, das Random-Search-Budget sei zu klein.
   **Diese Vermutung ist inzwischen widerlegt** (§ 12.7.1 / § 12.8): Weder
   der erschöpfende gridwh-Grid noch das schärfere PR-AUC-Kriterium (Option E,
   auf der RTX) wählen `window=48` oder heben `trend/wmz_1` über ≈ 0,61.
   Das Problem war ein **Kriteriums-Mismatch** (flache Val-ROC-AUC), kein
   Suchbudget — und auch dessen Behebung ändert die Rangfolge nicht.
   **Belastbare Schlussfolgerung für die Thesis:** Der LSTM-AE bleibt auf
   diesem einfachen, periodischen Regime confound-frei und kriteriums-robust
   unter der Klassik — kein methodisches Artefakt.
6. **WMZ-3 ist limitierend.** Mit nur 14 067 Trainings-Stunden und stark
   gefilterten Test-Stunden ist die statistische Aussagekraft gering. In
   der Thesis offen aussprechen und als Limitierung diskutieren (siehe
   Methodik 5.3, Memory `data-quality-wmz-glitches.md`).
7. **Stage 2 hat seinen Zweck erfüllt:** Die hohen Flag-Counts für
   `wmz_3` (416 031 Plateau-Flags!) sind kein Bug, sondern die korrekte
   Detektion der realen Sensorpathologie. Ohne diese Filterung würden alle
   späteren Modelle „Sensor-aus" als Normalmuster lernen und in der
   Konsequenz reale Heizphasen als Anomalie melden.
8. **Datenmengen-Sensitivität ist modellabhängig** (§ 13.1 + § 12.5):
   PELT ist invariant, IsolationForest sättigt bei ~10 000 Stunden,
   LOF hat ein nicht-monotones Optimum unter dem Maximum, und der
   LSTM-AE zeigt sogar **starkes Overfitting** bei voller Trainings­
   menge (auf `raw/wmz_1` ist F1 mit 2 000 Stunden = 0,765 vs. voller
   Menge = 0,212). Konsequenz: ein gemeinsam getuntes (HPs +
   Datenmenge)-Optimum wäre für den LSTM-AE noch zu suchen.

**Lesart für den Arbeitstitel („Muster- *und* Anomalieerkennung mit KI"):**
Die Mustererkennung leistet hier überwiegend die **klassische** MSTL-Zerlegung
(keine KI); die KI-Bausteine (LSTM-AE als Deep Learning, LOF/IF als
klassisches ML) erkennen Muster nur *implizit* — sie lernen das
Normalverhalten, um Abweichungen zu melden. Das „Klassik genügt"-Ergebnis ist
deshalb die gültige Antwort einer *Potenzialanalyse* und kein Widerspruch zum
Titel: gemessen wird, **wo** KI über die klassische Mustererkennung hinaus
Mehrwert bringt. Begriffliche Schärfung (KI weit/eng; zwei Muster-Ebenen) in
Gliederung § 2.0 und Methodik § 6.5; Reichweite des Befunds s. § 15/L3.

---

## 15 Limitationen und mögliche Ausbesserung

Die Ergebnisse sind im folgenden Rahmen zu lesen. Jede Limitation ist mit
einem konkreten Ansatz zur späteren Behebung verknüpft.

| # | Limitation | Mögliche Ausbesserung |
|---|---|---|
| L1 | **Einzelfallstudie (n = 1 Gebäude, 3 WMZ).** Die Befunde gelten zunächst nur für Gebäude A; Generalisierbarkeit ist nicht belegt. | **Zweiter Datensatz** über die `--dataset`-parametrisierte Pipeline (Stage 11, § 7.9) — kein neuer Code nötig. Verfügbare weitere WMZ-Datensätze haben **ähnliche Qualität** und sind **zeitlich kürzer** (Geb. A reicht am weitesten zurück) → sie prüfen **Generalisierung im selben Regime**, nicht das DL-Potenzial (gleiches Fazit zu erwarten; weniger Daten hilft DL nicht, vgl. § 13: LSTM-Overfitting bei voller Datenmenge). Seit 11/2025 existiert zudem ein **öffentliches, gelabeltes** DH-Substations-Dataset inkl. Fault-Detection-Evaluationsframework („PreDist", 547 Stationen; Roelofs u. a. 2026, *Energy*) — natürlicher Kandidat für Stage 11 mit **echten** Fault-Labels (adressiert zugleich L2). |
| L2 | **Synthetische Anomalien statt echter Labels.** Die quantitative Auswertung misst die *Wiedererkennung injizierter* Muster, nicht real validierte Fehler. | Label-freie Realwelt-Prüfung ausbauen (EnSikuMaV-Change-Points, § 10c); perspektivisch Experten-gelabelte reale Vorfälle. |
| L3 | **„Klassik genügt" gilt für ein einfaches, periodisches, homogenes Regime.** Keine generelle KI-Absage. | Heterogenere/gekoppelte Daten (mehrere Gebäude/Energieträger), wo Repräsentationslernen + multivariate AD greifen könnten (§ 7.10/§ 7.11). Die DL-Grenze ist in der Literatur kartiert (DL gewinnt auf komplexen, hochdim., multivariaten Benchmarks: Su u. a. 2019; Deng und Hooi 2021; Xu u. a. 2022; Schmidl u. a. 2022) → **analytisch** belegen. Ein **synthetischer** Datensatz nur als **ehrlicher Komplexitäts-Sweep** (ab welcher Stufe überholt DL?), **nicht** als „DL-gewinnt"-Konstruktion (zirkulär) → Ausblick. |
| L4 | ~~**Hardware-Confound im LSTM-Vergleich** (Mac-CPU-Diagnoselauf vs. CUDA-Baseline)~~ **ERLEDIGT (2026-06-10).** Der gridwh-Lauf wurde auf der RTX 3080 Ti (CUDA, `float32` wie alle LSTM-Läufe) wiederholt (§ 12.7.1) — gleiche Hardware wie die Baseline. *(Korrektur 2026-07-02: die Präzision war nie konfundiert — durchgängig `float32`.)* | **Aufgelöst:** Der confound-freie Lauf bestätigt die Kernaussage (LSTM-AE in allen 9 Jobs < Klassik-Champion). Auch Option E (PR-AUC-Kriterium, § 12.8) wurde durchgeführt — kein Effekt zugunsten `window=48`. |
| L5 | **Point-Adjust überschätzt** absolut (Kim u. a. 2022; Sehili und Zhang 2023). | Bereits abgefedert durch zusätzlich berichtete threshold-freie ROC-/PR-AUC; in der Diskussion explizit würdigen. |
| L6 | **LSTM-AE auf GPU nicht bit-reproduzierbar** (CUDA-Kernel). | ~~Seed-Averaging (Option E, `--hpo-seeds`) zur Varianz-Dämpfung; **offen:** die Inter-Seed-Streuung ist **nicht quantifiziert**.~~ **ERLEDIGT (2026-07-13, § 12.8.1 Pfad B):** Re-Lauf Option E mit `n_seeds = 3` und per-Seed-Logging quantifiziert die Inter-Seed-Std (Median 0,0058, Max 0,0298 auf trend/wmz_3). **`robust_2σ = 0/9`** — in keinem Job liegt der Best-vs-Runner-up-Abstand außerhalb des 2σ-Seed-Rauschens; die HP-Wahl ist statistisch beliebig (Reimers und Gurevych 2017; Bouthillier u. a. 2021). Deterministische Stages 1–6 + klassische Modelle bleiben exakt reproduzierbar. Das *stärkt* den Negativbefund gegen H2 nochmals: LSTM-AE-Selektion lebt in **keinem** Kriterium außerhalb des Init-Rauschens. |
| L7 | **Kein empirischer Auflösungsvergleich (H5).** Die Pipeline läuft durchgängig stündlich; H5 wird **analytisch** beantwortet (thermisch träge B-Anomalien → stündlich angemessen; sub-stündliche Ereignisse sind Sensorfehler der Kat. A, regelbasiert gefiltert). Belege: Frederiksen und Werner 2013; Lindberg u. a. 2019; Katipamula und Brambley 2005; Amirkhanova u. a. 2026; Himeur u. a. 2021; EN 1434 2015. | Gezielter Minuten-Lauf auf **Variante A** mit **echter sub-stündlicher Fehlersignatur** (Oszillation/Hunting), sobald gelabelte sub-stündliche Vorfälle vorliegen — verkürzte Spikes zeigen nur den Mittelungseffekt. |
| L8 | **Datenqualität/Interpolation als möglicher (nicht isolierter) Mitgrund der LSTM-AE-Schwäche.** Stage 2 interpoliert kurze Lücken (Anteil interpolierter Stunden: wmz_1 15,9 %, wmz_2 5,1 %, **wmz_3 38,6 %**). Der LSTM-AE verwirft NaN-Fenster (`_make_windows`) → auf B/C schrumpft die ohnehin knappe Trainingsmenge, auf A lernt er interpolierte lineare Rampen als „normal". Punktweise Klassik (Z/LOF/IF) und Trend-PELT sind dagegen robuster. | **Nicht als Hauptursache belegt:** Keine Dosis-Wirkung (LSTM-AE ist am *besten* auf wmz_1 mit 16 % und am *schlechtesten* auf wmz_2 mit nur 5 %); die Fehlersignatur ist **Score-Inversion (ROC-AUC < 0,5)**, kein Fehlende-Daten-Effekt. Ablation zur Isolierung: LSTM-AE nur auf interpolationsfreien Fenstern re-evaluieren bzw. Interpolationsanteil je Fenster ↔ Rekonstruktionsfehler korrelieren. |

**Objektivitäts-Schritt (durchgeführt 2026-06-10):** Der gridwh-Lauf wurde auf
der **RTX 3080 Ti** (CUDA, `float32`) wiederholt und löst den
Hardware-Confound (L4) auf — Ergebnisse in § 12.7.1 (gridwh-GPU) und
§ 12.8 (Option E, PR-AUC-Kriterium). Die LSTM-AE-Aussage steht damit
confound-frei und kriteriums-robust: in **keinem** der 9 Jobs wird der
Klassik-Champion geschlagen. **Nachtrag 2026-07-13 (§ 12.8.1):** Auch das
Mehr-Seed-Averaging (Pfad B) und ein dritter Selektor (F1-HPO, Pfad C)
sind jetzt durchgeführt — beides bestätigt den Negativbefund. L6 gilt
damit als **empirisch belegt** (`robust_2σ = 0/9`, HP-Wahl im
Init-Rauschen); der F1-Selektor wählt in 8/9 Jobs andere Configs als
PR-AUC, ohne den Test-F1-Abstand zur Klassik zu schließen.
**Nachtrag 2026-07-20 (§ 12.8.2):** Die Score-Ablation
(`window_mse`, `channel_mse`, `last_step_mse`, `mahalanobis`
nach Malhotra u. a. 2016) ist jetzt ebenfalls auf der RTX
durchgeführt — Score-Reparatur ist in **3 von 9 Jobs** empirisch belegt
(Δ F1 bis +0,689 unter `last_step_mse` auf residual/wmz_1); in **6 von 9**
bleibt die LSTM-AE-Schwäche unter allen vier Modi bestehen und der
Klassik-Champion wird in **keinem** Job unter irgendeinem Scoring
geschlagen. Die Ebene-4-Diagnose ist damit quantifiziert (Kap. 5.6).
Die GPU-Artefakte (best_hparams + Metriken für gridwh-GPU, Option E,
Option-E-Seedstd, Option-E-F1 und Baseline) sind in
`outputs/gebaeude_a/{hpo,reports}/` gespiegelt; der aktuelle
Schreib-Export `export_schreiben_2026-07-13.zip` bündelt alle Stände.
Die Haupt-`stage10_metrics.csv` / `best_hparams.json` zeigen weiterhin
die Random-Search-Baseline (= § 11.2); ältere Zwischenstände sind als
`*_gridwh_cpu.*` bzw. `*_pre_seedfix.*` archiviert.

---

## 16 Vollständige Artefaktliste

| Artefakt | Pfad |
|---|---|
| Stage-1 Roh-Merge | [parquet/stage1_raw_merged.parquet](../outputs/gebaeude_a/parquet/stage1_raw_merged.parquet) (+ CSV) |
| Stage-2 stündlich | [parquet/stage2_hourly.parquet](../outputs/gebaeude_a/parquet/stage2_hourly.parquet) (+ CSV) |
| Stage-2 Flag-Log | [reports/stage2_flag_log.csv](../outputs/gebaeude_a/reports/stage2_flag_log.csv) |
| Stage-2 Gap-Log | [reports/stage2_gap_log.csv](../outputs/gebaeude_a/reports/stage2_gap_log.csv) |
| Stage-2 Glitch-Log | [reports/stage2_glitch_log.csv](../outputs/gebaeude_a/reports/stage2_glitch_log.csv) |
| Stage-2 Reset-Log | [reports/stage2_reset_log.csv](../outputs/gebaeude_a/reports/stage2_reset_log.csv) |
| Stage-3 STL-Zerlegung | [parquet/stage3_stl.parquet](../outputs/gebaeude_a/parquet/stage3_stl.parquet) |
| Stage-3 Figures | [figures/stl_*.png](../outputs/gebaeude_a/figures/), [figures/heizsaison_vergleich.png](../outputs/gebaeude_a/figures/heizsaison_vergleich.png) |
| Stage-4 Variante A | [parquet/stage4_features_raw.parquet](../outputs/gebaeude_a/parquet/stage4_features_raw.parquet) |
| Stage-4 Variante B | [parquet/stage4_features_residual.parquet](../outputs/gebaeude_a/parquet/stage4_features_residual.parquet) |
| Stage-5 Split | [parquet/split_assignment.parquet](../outputs/gebaeude_a/parquet/split_assignment.parquet) |
| Stage-6 Normalisierte Frames | [parquet/stage6_normalized_{raw,residual,trend}.parquet](../outputs/gebaeude_a/parquet/) |
| Stage-6 Scaler | [scalers/scaler_{raw,residual,trend}.parquet](../outputs/gebaeude_a/scalers/) |
| Stage-7 Modelle | [models/{raw,residual,trend}/<wmz>/<modell>.pkl](../outputs/gebaeude_a/models/) (21 Dateien) |
| Stage-8 Beste HPs | [hpo/best_hparams.json](../outputs/gebaeude_a/hpo/best_hparams.json) |
| Stage-8 Voller HPO-Log | [hpo/hpo_log.csv](../outputs/gebaeude_a/hpo/hpo_log.csv) |
| Stage-9 Injizierte Reihen | [parquet/stage9_injected_{raw,residual,trend}.parquet](../outputs/gebaeude_a/parquet/) |
| Stage-9 Ground-Truth | [parquet/stage9_ground_truth.parquet](../outputs/gebaeude_a/parquet/stage9_ground_truth.parquet) |
| Stage-10 Metriken (Haupt = § 11.2-Baseline) | [reports/stage10_metrics.csv](../outputs/gebaeude_a/reports/stage10_metrics.csv) |
| Stage-10 Change-Points | [reports/stage10_changepoints_regulatory.csv](../outputs/gebaeude_a/reports/stage10_changepoints_regulatory.csv) |
| Stage-10 Qualitativ | [reports/stage10_qualitative_*.csv](../outputs/gebaeude_a/reports/) (21 Dateien) |
| Pipeline-Konsolen-Logs | `outputs/stage{8,9,10}_log.txt` |
| **LSTM-GPU-Reruns (2026-06-10)** — HPs | `hpo/best_hparams_{before_gpu_nightrun,gridwh_gpu_roc_auc,option_e_pr_auc_gpu}.json` |
| LSTM-GPU-Reruns — Metriken | `reports/stage10_metrics_{before_gpu_nightrun,gridwh_gpu_roc_auc,option_e_pr_auc_gpu}.csv` |
| LSTM-GPU-Reruns — HPO-Logs | `hpo/hpo_log_{before_gpu_nightrun,gridwh_gpu_roc_auc,option_e_pr_auc_gpu}.csv` |
| LSTM-Seed-Robustheit (§ 12.8 / L6) | [reports/lstm_seed_variance.csv](../outputs/gebaeude_a/reports/lstm_seed_variance.csv) |
| **LSTM-AE Score-Ablation** (§ 12.8.2) | [reports/lstm_ae_score_ablation.csv](../outputs/gebaeude_a/reports/lstm_ae_score_ablation.csv) |
| gridwh-**CPU**-Diagnose (§ 12.7, archiviert) | `reports/stage10_metrics_gridwh_cpu.csv`, `hpo/best_hparams_gridwh_cpu.json` |

---

## 17 Anhang A — LSTM-Autoencoder Smoke-Test (historisch)

> **Hinweis (Stand 2026-06-02):** Diese Sektion dokumentiert den
> ursprünglichen Smoke-Test mit reduziertem Budget. Die finalen
> LSTM-AE-Ergebnisse mit vollem HPO sind in **§ 12** ab Zeile
> „15 LSTM-Autoencoder — Vollständige HPO-Ergebnisse" zu finden.
> Die Smoke-Sektion bleibt als methodischer Verlauf erhalten
> (Pipeline-Verifikation, Daten-Sweep auf Smoke-Konfigurationen).

Diese Sektion dokumentiert den **eingeschränkten Vorab-Lauf** des LSTM-AE
auf der RTX 3080 Ti Mobile. Das vollständige HPO-Budget ist in
§ 12 als Platzhalter mit konkreter Befehlsfolge dokumentiert und
wird in einer separaten Nacht-Session durchgeführt.

> **Hinweis zur Lesart:** Die Zahlen in dieser Sektion sind ein
> Funktionsnachweis der GPU-Pipeline, **kein finales Ergebnis**. Sowohl
> die Trainingsdaten als auch das Random-Search-Budget wurden gegenüber
> dem Soll-Lauf stark reduziert (siehe 14.2). Belastbare Vergleiche zu
> den klassischen Modellen sind erst mit dem vollen HPO-Lauf möglich.

### 17.1 Setup

* **Hardware:** NVIDIA RTX 3080 Ti Mobile, 16 GB VRAM, CUDA 12.4
* **Framework:** torch 2.6.0+cu124 (Installation via
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
* **Modellarchitektur:** Sequenz-Autoencoder mit gestapeltem LSTM
  (Encoder verdichtet ein Fenster auf einen Latent-Vektor, Decoder
  rekonstruiert die gesamte Sequenz); Score = MSE zwischen Eingabe-
  und Rekonstruktions-Fenster
* **Anomalie-Score-Verankerung:** Ein Score wird der **letzten Stunde
  jedes Fensters** zugewiesen; die ersten `window_size − 1` Stunden
  und Fenster mit NaN bekommen NaN-Scores

### 17.2 Smoke-Test-Konfiguration (reduziertes Budget)

| Parameter | Sollwert (volles HPO) | Smoke-Wert | Begründung |
|---|---|---|---|
| `--lstm-search-iter` | 25 | **3** | Soll nur die Pipeline-Integrität verifizieren |
| `--max-train-rows` | (unbegrenzt) | **5 000** | Train-Zeit pro fit etwa um Faktor 5 verkürzt |
| Anzahl LSTM-Jobs | 9 (3 Varianten × 3 WMZ) | 9 | unverändert |
| HP-Suchraum | `window ∈ {24,48,72}`, `hidden ∈ {16,32,64}`, `layers ∈ {1,2}`, `lr ∈ {1e−2,3e−3,1e−3,3e−4}`, `epochs ∈ {30,50,100}` | identisch | nur die Anzahl gezogener Stichproben ist reduziert |

### 17.3 Pipeline-Performance auf der RTX 3080 Ti Mobile

| Stage | Modus | Anzahl fits | Wandzeit | ø pro fit |
|---|---|---:|---:|---:|
| Stage 7 LSTM-AE | volle Train-Daten, Default-HPs | 9 | **7 min 10 s** | ~48 s |
| Stage 8 LSTM-AE Smoke | 3 Configs × 9 Jobs × `max-train-rows=5 000` | 27 | **3 min 49 s** | ~8 s |
| Stage 10 LSTM-AE Smoke | beste Smoke-HPs, `max-train-rows=5 000` | 9 | **1 min 26 s** | ~10 s |
| **Summe Smoke** | | 45 | **~12 min 25 s** | |

**Hochrechnung auf das volle HPO** (§ 12): bei 25 Configs × 9 Jobs
= 225 fits und voller Train-Datenmenge erwarten wir nach dem
Smoke-Verhältnis (Faktor 5 für die Trainingsdaten × Faktor 8 für die
Anzahl Configs) eine Stage-8-Dauer von **~10–14 Stunden** — konsistent
zur Vorab-Schätzung. Stage 7 ist bereits voll gelaufen und muss vor dem
HPO **nicht** wiederholt werden.

### 17.4 Smoke-Test: beste HPs (nach 3 Random-Search-Stichproben)

Die folgenden HPs sind **nicht** belastbar — bei nur 3 gezogenen
Konfigurationen pro Job ist die Reihenfolge weitgehend zufällig. Sie
dienen ausschließlich der Smoke-Eval.

| Variante | WMZ | beste Smoke-HP | Val-AUC |
|---|---|---|---:|
| raw | wmz_1 | window=48, hidden=32, layers=2, lr=3e−3, epochs=50 | 0,487 |
| raw | wmz_2 | window=72, hidden=64, layers=1, lr=1e−2, epochs=50 | 0,446 |
| raw | wmz_3 | window=72, hidden=64, layers=1, lr=1e−2, epochs=50 | 0,454 |
| residual | wmz_1 | window=48, hidden=32, layers=2, lr=3e−3, epochs=50 | 0,547 |
| residual | wmz_2 | window=72, hidden=64, layers=1, lr=1e−2, epochs=50 | 0,470 |
| residual | wmz_3 | window=48, hidden=64, layers=1, lr=1e−2, epochs=30 | 0,425 |
| trend | wmz_1 | window=72, hidden=64, layers=1, lr=1e−2, epochs=50 | 0,518 |
| trend | wmz_2 | window=48, hidden=64, layers=1, lr=1e−2, epochs=30 | 0,619 |
| trend | wmz_3 | window=48, hidden=32, layers=2, lr=3e−3, epochs=50 | 0,663 |

### 17.5 Smoke-Test: Metriken auf dem Test-Set

| Variante | WMZ | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| raw | wmz_1 | 0,009 | 0,005 | 0,006 | 0,599 |
| raw | wmz_2 | 0,000 | 0,000 | 0,000 | 0,409 |
| raw | wmz_3 | 0,000 | 0,000 | 0,000 | 0,391 |
| residual | wmz_1 | 0,013 | 0,008 | 0,010 | 0,637 |
| residual | wmz_2 | 0,000 | 0,000 | 0,000 | 0,458 |
| residual | wmz_3 | 0,000 | 0,000 | 0,000 | 0,416 |
| trend | wmz_1 | **0,714** | **0,441** | **0,545** | 0,257 |
| trend | wmz_2 | 0,000 | 0,000 | 0,000 | 0,447 |
| trend | wmz_3 | 0,000 | 0,000 | 0,000 | 0,529 |

**Aggregierte F1 (Mittel über WMZ):**

| Variante | Ø F1 (LSTM-Smoke) | zum Vergleich Klassik-Ø F1 |
|---|---:|---:|
| raw | 0,002 | 0,313 |
| residual | 0,003 | 0,215 |
| trend | 0,182 | 0,410 |

### 17.6 Interpretation der Smoke-Ergebnisse

* **Erwartungsgemäß schwach im Gesamtbild:** Mit nur 5 000 Trainings-
  Stunden (statt 14 067–24 414) und nur 3 Random-Search-Konfigurationen
  (statt 25) hat der LSTM-AE keine realistische Chance, Saisonalität,
  Wochenstruktur und Wetterabhängigkeit verlässlich zu lernen. AUC-Werte
  knapp unter 0,5 bei `wmz_2`/`wmz_3` deuten darauf hin, dass der Score
  in den meisten Fällen schlechter als zufällig ist — typisches Verhalten
  eines untertrainierten Autoencoders.
* **Positiver Ausreißer: trend/wmz_1 (F1 = 0,545).** Selbst mit Smoke-HPs
  und reduziertem Train-Set findet der LSTM-AE auf der univariaten
  Trend-Reihe Strukturen, die zu Drift-Anomalien passen. Das spricht
  dafür, dass die Architektur und die Pipeline funktionieren — und macht
  ein vollständiges HPO besonders aussichtsreich.
* **Pipeline-Integrität bestätigt:** CUDA wird korrekt erkannt
  (`device='cuda'`), Train/Score/Save/Load der `.pkl`-Dateien laufen
  fehlerfrei, das gemeinsame `best_hparams.json` mischt sich sauber mit
  den Klassik-HPs, und Stage 10 nutzt für jedes Modell die korrekten HPs.
  **Der Soll-Lauf kann ohne weitere Code-Änderungen gestartet werden.**

### 17.7 Smoke-Artefakte

Die Smoke-Outputs wurden **zusätzlich** zu den Klassik-Outputs persistiert,
damit der nächtliche volle HPO-Lauf die Smoke-Resultate nicht stillschweigend
überschreibt (Stage 8 / Stage 10 schreiben jeweils die Gesamt-Datei neu):

| Artefakt | Pfad |
|---|---|
| Smoke-HPs (LSTM-AE only) | [best_hparams_lstm_smoke.json](../outputs/_archiv/snapshots_zwischenstaende/best_hparams_lstm_smoke.json) (seit 2026-07-02 in `outputs/_archiv/`) |
| Smoke-HPO-Log (LSTM-AE only) | [hpo_log_lstm_smoke.csv](../outputs/_archiv/snapshots_zwischenstaende/hpo_log_lstm_smoke.csv) (Archiv) |
| Klassik-Metriken (vor LSTM-Eval gesichert) | [reports/stage10_metrics_classic.csv](../outputs/gebaeude_a/reports/stage10_metrics_classic.csv) |
| Gemerged: alle 5 Modelle | [reports/stage10_metrics.csv](../outputs/gebaeude_a/reports/stage10_metrics.csv) (322 Zeilen; Stand vor Constancy — heute 357 Zeilen / 6 Modelle) |
| Smoke-Logs | `outputs/_archiv/logs/stage{7_lstm,8_lstm_smoke,10_lstm_smoke}_log.txt` |
| Stage-10 qualitative TopN (LSTM) | [reports/stage10_qualitative_*_lstm_ae.csv](../outputs/gebaeude_a/reports/) (9 Dateien) |

---

## 18 Anhang B — Robustheit der Schwellenparameter (Sensitivitätsanalyse)

Der kW-Spike-Detektor (`is_kw_spike`, § 2.1) nutzt zwei frei gewählte
Domänen-Konstanten: den **Faktor** (lokaler Median × 10) und den **Sockel**
(+ 5 kW). Um zu belegen, dass die bereinigte Datenbasis nicht von der konkreten
Wahl dieser Werte abhängt, wurde jeweils eine Konstante variiert und die andere
fixiert. Gemessen wird die **Zahl der geflaggten Stunden je WMZ** — die für die
Bereinigung relevante Endgröße, nicht die rohe Minuten-Flag-Zahl. Reproduziert
mit den echten Stage-2-Detektoren auf der vollen Minutenreihe (bei Sockel 5 /
Faktor 10 reproduzieren die Werte exakt die § 2.2-Zahlen).

### 18.1 Sockel (Faktor fix = 10)

| WMZ | Sockel 0 kW | 5 kW | 10 kW | Spanne |
|---|---:|---:|---:|---:|
| wmz_1 | 5 563 | **5 563** | 5 563 | 0 |
| wmz_2 | 1 786 | **1 780** | 1 772 | 14 |
| wmz_3 | 18 225 | **18 225** | 18 225 | 0 |

Nachrichtlich `is_kw_spike` (Minuten) bei wmz_2: 145 / 102 / 68.

**Befund:** wmz_1 und wmz_3 sind vollständig invariant. wmz_2 verschiebt sich
über die gesamte Spanne 0–10 kW um nur **14 von 35 064 Stunden** (0,04 %). Der
Sockel ist damit nicht ergebnisbestimmend; der Wert 5 kW entspricht der
Größenordnung der zählerseitigen Grundlast.

### 18.2 Faktor (Sockel fix = 5 kW)

| WMZ | Faktor 5 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|
| wmz_1 | 5 568 | 5 563 | **5 563** | 5 561 | 5 561 |
| wmz_2 | 1 918 | 1 813 | **1 780** | 1 763 | 1 762 |
| wmz_3 | 18 225 | 18 225 | **18 225** | 18 225 | 18 225 |

Nachrichtlich `is_kw_spike` (Minuten) bei wmz_2: 501 / 189 / 102 / 63 / 60.

**Befund:** wmz_1 (Spanne 7 Stunden) und wmz_3 (0) sind robust. wmz_2 ist am
unteren Ende sensitiv — Faktor 5 über-flaggt die natürliche Warmwasser-
Volatilität (+138 Stunden gegenüber Faktor 10) —, läuft aber ab Faktor 10 in
einen stabilen Bereich ein (Grenzänderung ≤ 17 Stunden je Stufe: 10→15 = −17,
15→20 = −1). Der gewählte Faktor 10 liegt damit am **Knie der
Sensitivitätskurve**: groß genug, um legitime Volatilität nicht mehr zu flaggen,
klein genug, um echte Spikes zu erfassen.

### 18.3 Fazit

Die bereinigte Datenbasis ist gegenüber dem Sockel praktisch invariant und
gegenüber dem Faktor ab dem gewählten Wert stabil. Die beiden Konstanten
(Faktor 10, Sockel 5 kW) sind damit **zweckmäßig begründet, aber nicht
ergebnisbestimmend** — die in § 2.2 berichteten Flag-Zahlen und die
nachfolgenden Stages hängen nicht an ihrer exakten Wahl. Reproduktion: Variation
von `KW_SPIKE_FACTOR_OVER_MEDIAN` bzw. `KW_SPIKE_SOCKET` in
[stage2_preprocess.py](../src/stage2_preprocess.py#L72-L73), Aggregation der
Flags auf das Stundenraster.

---

