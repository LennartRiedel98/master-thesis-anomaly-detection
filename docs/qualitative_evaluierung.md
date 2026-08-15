# Qualitative Evaluierung — Fall-Protokoll

> **Hinweis zur veröffentlichten Fassung.** Grundlage sind die **realen
> Betriebsdaten**, die dem Repository nicht beiliegen (siehe
> [README, Abschnitt „Daten"](../README.md#daten)); Gebäude, Betreiber,
> Standort und Zählernummern sind anonymisiert.

*Automatisch erzeugt von [src/tools/qualitative_protocol.py](../src/tools/qualitative_protocol.py).*
Datensatz: `gebaeude_a`. Schwelle: 99%-Quantil der sauberen Validierungs-Scores (PELT binär bei 0.5), Stage-10-identisch. Einzelfälle visuell: `plot_qualitative_case.py` (siehe Plot-Befehle unten).

Die faktischen Teile (A, A2, B, C — Recall/Urteil/Score/FP) sind datengetrieben; die **Interpretation** (Teil D, Spalten Beobachtung/Erklärung) ist bewusst manuell auszufüllen — sie ist der eigentliche wissenschaftliche Beitrag, nicht automatisierbar ohne Fabrikation.

---

## A. Aggregierte Detektions-Matrix (Recall je Typ × Detektor)

Quelle: `stage10_metrics.csv` (Point-adjusted Recall, über alle drei WMZ summiert: erkannte / injizierte Events). Enthält **alle** Modelle inkl. LSTM-AE — exakt die im Ergebnisbericht berichteten Zahlen.

| Detektor | spike | drop | plateau | leakage | drift | structural_break |
|---|---|---|---|---|---|---|
| **zscore** | 20/38 (53%) | 13/30 (43%) | 0/24 (0%) | 7/48 (15%) | – | – |
| **lof** | 26/38 (68%) | 1/30 (3%) | 1/24 (4%) | 16/48 (33%) | – | – |
| **iforest** | 10/38 (26%) | 4/30 (13%) | 0/24 (0%) | 10/48 (21%) | – | – |
| **constancy** | 0/19 (0%) | 0/15 (0%) | 8/12 (67%) | 9/24 (38%) | – | – |
| **pelt** | – | – | – | – | 6/23 (26%) | 1/14 (7%) |
| **lstm_ae** | 18/38 (47%) | 0/30 (0%) | 0/24 (0%) | 1/48 (2%) | 5/23 (22%) | 4/14 (29%) |

*Lesart:* `erkannt/injiziert (Recall)`. „–" = Detektor läuft nicht auf der Schiene dieses Typs (z. B. PELT nur auf Trend → nur drift/structural_break; Constancy nur auf raw).

*Hinweis zu den Nennern:* zscore/lof/iforest laufen auf **zwei** Schienen (raw **und** residual), ihre injizierten Events zählen daher doppelt (z. B. spike: 2 Varianten × 3 WMZ); constancy (nur raw) und pelt (nur trend) haben entsprechend kleinere Nenner. Der Recall ist je Detektor korrekt, die absoluten Event-Zahlen sind zwischen den Detektoren also **nicht** 1:1 vergleichbar — die schienenscharfe Aufschlüsselung liefert Teil B.

---

## A2. Abdeckungsanalyse — Union-Recall der Detektor-Suite

Pro injiziertem Event: erkennt es **mindestens ein** Detektor (logisches ODER über alle Detektoren der zuständigen Schiene)? Zeigt, ob die komplementäre Suite gemeinsam jeden Typ abdeckt — die eigentliche Aussage hinter dem Mehr-Detektoren-Ansatz. Vergleich: bestes Einzelmodell vs. Union.

| Typ | Events | Bestes Einzelmodell | Union (≥1 Detektor) | Zugewinn |
|---|---|---|---|---|
| spike | 36 | lof: 28/36 (78%) | 29/36 (**81%**) | +3% |
| drop | 24 | zscore: 8/24 (33%) | 10/24 (**42%**) | +8% |
| plateau | 12 | constancy: 8/12 (67%) | 8/12 (**67%**) | +0% |
| leakage | 24 | lof: 10/24 (42%) | 17/24 (**71%**) | +29% |
| drift | 24 | pelt: 10/24 (42%) | 10/24 (**42%**) | +0% |
| structural_break | 24 | pelt: 1/24 (4%) | 1/24 (**4%**) | +0% |

*Lesart:* „Union" = Anteil Events, die **irgendein** Detektor der Schiene fängt. Ein großer Zugewinn gegenüber dem besten Einzelmodell belegt, dass sich die Detektoren ergänzen (kein einzelnes Modell deckt alle Typen ab). Hinweis: Die Union maximiert den Recall, erhöht aber auch die Fehlalarmrate (Union der False Positives) — die Präzisionsseite ist gesondert zu bewerten. **Achtung:** ohne `--include-lstm` zählt der LSTM-AE hier nicht zur Union — besonders relevant für drift/structural_break, wo er der einzige Ko-Detektor neben PELT ist; die Trend-Union ist dann unterschätzt.

---

## B. Event-Detektions-Matrix je WMZ

Eine Zeile pro injiziertem Event. ✓ = als TP erkannt (irgendein Punkt im Event über Schwelle, point-adjusted), ✗ = verpasst (FN). Der Wert in Klammern ist der **Spitzen-Score im Event ÷ Schwelle** — Werte knapp unter 1.00 markieren „fast erkannt".

### raw / wmz_1  (Schiene: gt_stat)

| # | Event | Start | Dauer | zscore | lof | iforest | constancy |
|---|---|---|---|---|---|---|---|
| 1 | leakage@0.15 | 2023-05-03 06:00 | 146 h | ✓ (1.03) | ✓ (1.00) | ✓ (1.03) | ✓ (1.02) |
| 2 | spike@5 | 2023-05-12 09:00 | 1 h | ✓ (6.59) | ✓ (4.98) | ✓ (1.05) | ✗ (0.00) |
| 3 | spike@5 | 2023-05-19 21:00 | 1 h | ✓ (4.96) | ✓ (3.54) | ✓ (1.05) | ✗ (0.00) |
| 4 | drop@0 | 2023-06-03 07:00 | 2 h | ✓ (1.68) | ✗ (0.52) | ✓ (1.06) | ✗ (0.00) |
| 5 | spike@3 | 2023-06-07 14:00 | 1 h | ✓ (3.22) | ✓ (2.01) | ✓ (1.06) | ✗ (0.00) |
| 6 | spike@10 | 2023-06-10 11:00 | 1 h | ✓ (37.66) | ✓ (28.83) | ✓ (1.10) | ✗ (0.00) |
| 7 | drop@0.25 | 2023-06-13 20:00 | 1 h | ✓ (1.17) | ✗ (0.56) | ✓ (1.00) | ✗ (0.00) |
| 8 | plateau | 2023-06-18 11:00 | 10 h | ✗ (0.51) | ✗ (0.59) | ✗ (0.86) | ✓ (1.08) |
| 9 | plateau | 2023-06-23 23:00 | 15 h | ✗ (0.53) | ✗ (0.92) | ✗ (0.93) | ✓ (1.08) |
| 10 | leakage@0.3 | 2023-06-27 23:00 | 138 h | ✓ (1.15) | ✓ (1.34) | ✓ (1.04) | ✗ (0.23) |
| 11 | spike@3 | 2023-07-06 07:00 | 1 h | ✓ (3.37) | ✓ (2.13) | ✓ (1.03) | ✗ (0.00) |
| 12 | leakage@0.15 | 2023-07-14 14:00 | 132 h | ✓ (1.13) | ✓ (1.30) | ✓ (1.05) | ✗ (0.61) |
| 13 | spike@10 | 2023-07-22 17:00 | 1 h | ✓ (11.15) | ✓ (8.30) | ✓ (1.09) | ✗ (0.00) |
| 14 | spike@5 | 2023-07-30 01:00 | 1 h | ✓ (5.14) | ✓ (3.32) | ✓ (1.08) | ✗ (0.00) |
| 15 | spike@5 | 2023-08-05 05:00 | 1 h | ✓ (4.96) | ✓ (3.49) | ✓ (1.09) | ✗ (0.00) |
| 16 | drop@0.25 | 2023-08-08 18:00 | 3 h | ✓ (1.44) | ✗ (0.52) | ✓ (1.02) | ✗ (0.00) |
| 17 | leakage@0.3 | 2023-08-12 05:00 | 160 h | ✓ (1.33) | ✓ (1.26) | ✓ (1.06) | ✗ (0.46) |
| 18 | spike@3 | 2023-08-23 12:00 | 1 h | ✓ (3.10) | ✓ (1.85) | ✓ (1.05) | ✗ (0.00) |
| 19 | leakage@0.3 | 2023-08-31 17:00 | 157 h | ✓ (1.13) | ✓ (1.28) | ✓ (1.05) | ✗ (0.85) |
| 20 | drop@0.25 | 2023-09-10 06:00 | 1 h | ✓ (1.16) | ✗ (0.63) | ✓ (1.04) | ✗ (0.00) |
| 21 | drop@0 | 2023-09-14 23:00 | 1 h | ✓ (1.43) | ✗ (0.48) | ✓ (1.04) | ✗ (0.00) |
| 22 | leakage@0.3 | 2023-09-17 23:00 | 63 h | ✓ (1.19) | ✓ (1.16) | ✓ (1.04) | ✗ (0.48) |
| 23 | leakage@0.15 | 2023-09-24 23:00 | 48 h | ✓ (1.07) | ✓ (1.09) | ✓ (1.06) | ✗ (0.66) |
| 24 | spike@10 | 2023-09-30 08:00 | 1 h | ✓ (11.77) | ✓ (8.83) | ✓ (1.08) | ✗ (0.00) |
| 25 | drop@0.25 | 2023-10-06 20:00 | 3 h | ✓ (1.40) | ✗ (0.62) | ✓ (1.07) | ✗ (0.00) |
| 26 | spike@10 | 2023-10-10 10:00 | 1 h | ✓ (11.25) | ✓ (8.38) | ✓ (1.05) | ✗ (0.00) |
| 27 | leakage@0.15 | 2023-10-17 11:00 | 93 h | ✓ (1.16) | ✗ (0.97) | ✓ (1.05) | ✗ (0.64) |
| 28 | plateau | 2023-10-24 01:00 | 11 h | ✗ (0.49) | ✗ (0.83) | ✗ (0.86) | ✓ (1.08) |
| 29 | drop@0 | 2023-10-28 23:00 | 2 h | ✓ (1.69) | ✗ (0.52) | ✓ (1.08) | ✗ (0.00) |
| 30 | drop@0 | 2023-11-01 11:00 | 2 h | ✓ (1.66) | ✗ (0.53) | ✓ (1.03) | ✗ (0.00) |
| 31 | spike@3 | 2023-11-08 10:00 | 1 h | ✓ (2.88) | ✓ (1.77) | ✓ (1.03) | ✗ (0.00) |
| 32 | plateau | 2023-11-15 06:00 | 16 h | ✗ (0.60) | ✗ (0.71) | ✗ (0.87) | ✓ (1.08) |

### raw / wmz_2  (Schiene: gt_stat)

| # | Event | Start | Dauer | zscore | lof | iforest | constancy |
|---|---|---|---|---|---|---|---|
| 1 | spike@10 | 2023-05-05 17:00 | 1 h | ✗ (0.54) | ✓ (1.22) | ✗ (0.94) | ✗ (0.00) |
| 2 | spike@10 | 2023-05-09 13:00 | 1 h | ✓ (2.94) | ✓ (3.26) | ✓ (1.08) | ✗ (0.00) |
| 3 | spike@3 | 2023-05-13 23:00 | 1 h | ✗ (0.27) | ✗ (1.00) | ✗ (0.89) | ✗ (0.00) |
| 4 | spike@5 | 2023-05-18 01:00 | 1 h | ✗ (1.00) | ✗ (0.88) | ✗ (0.89) | ✗ (0.00) |
| 5 | spike@10 | 2023-05-25 09:00 | 1 h | ✗ (0.88) | ✓ (1.24) | ✗ (0.99) | ✗ (0.00) |
| 6 | plateau | 2023-05-28 19:00 | 21 h | ✗ (1.00) | ✗ (0.63) | ✗ (0.88) | ✓ (1.12) |
| 7 | leakage@0.15 | 2023-06-06 11:00 | 65 h | ✗ (0.48) | ✗ (0.93) | ✗ (0.85) | ✓ (1.09) |
| 8 | plateau | 2023-06-13 22:00 | 15 h | ✗ (0.31) | ✗ (0.65) | ✗ (0.77) | ✓ (1.12) |
| 9 | drop@0 | 2023-06-22 10:00 | 2 h | ✗ (0.29) | ✗ (0.76) | ✗ (0.80) | ✗ (0.00) |
| 10 | leakage@0.3 | 2023-06-28 23:00 | 111 h | ✗ (0.30) | ✗ (0.77) | ✗ (0.82) | ✓ (1.05) |
| 11 | leakage@0.3 | 2023-07-05 20:00 | 75 h | ✗ (0.47) | ✗ (0.89) | ✗ (0.86) | ✓ (1.06) |
| 12 | plateau | 2023-07-14 03:00 | 9 h | ✗ (0.25) | ✗ (0.60) | ✗ (0.78) | ✓ (1.12) |
| 13 | drop@0 | 2023-07-17 07:00 | 2 h | ✗ (0.25) | ✗ (0.74) | ✗ (0.81) | ✗ (0.02) |
| 14 | spike@3 | 2023-07-22 19:00 | 1 h | ✗ (0.35) | ✗ (0.82) | ✗ (0.89) | ✗ (0.00) |
| 15 | leakage@0.3 | 2023-07-25 18:00 | 54 h | ✗ (0.33) | ✗ (0.98) | ✗ (0.82) | ✗ (0.98) |
| 16 | drop@0.25 | 2023-07-31 07:00 | 3 h | ✗ (0.25) | ✗ (0.68) | ✗ (0.79) | ✗ (0.04) |
| 17 | spike@10 | 2023-08-09 12:00 | 1 h | ✓ (1.79) | ✓ (1.72) | ✓ (1.03) | ✗ (0.00) |
| 18 | drop@0.25 | 2023-08-14 23:00 | 3 h | ✗ (0.25) | ✗ (0.73) | ✗ (0.79) | ✗ (0.09) |
| 19 | leakage@0.3 | 2023-08-17 16:00 | 66 h | ✗ (0.43) | ✗ (0.75) | ✗ (0.82) | ✓ (1.08) |
| 20 | spike@3 | 2023-08-24 14:00 | 1 h | ✗ (0.54) | ✓ (1.44) | ✗ (0.91) | ✗ (0.00) |
| 21 | spike@5 | 2023-08-28 22:00 | 1 h | ✗ (0.25) | ✓ (1.51) | ✗ (0.87) | ✗ (0.00) |
| 22 | drop@0.25 | 2023-09-07 08:00 | 2 h | ✗ (0.22) | ✗ (0.72) | ✗ (0.77) | ✗ (0.01) |
| 23 | leakage@0.15 | 2023-09-11 11:00 | 73 h | ✗ (0.42) | ✗ (0.90) | ✗ (0.85) | ✓ (1.07) |
| 24 | plateau | 2023-09-21 00:00 | 16 h | ✗ (0.37) | ✗ (0.78) | ✗ (0.83) | ✓ (1.12) |
| 25 | drop@0.25 | 2023-09-27 08:00 | 3 h | ✗ (0.24) | ✗ (0.83) | ✗ (0.79) | ✗ (0.00) |
| 26 | spike@3 | 2023-10-05 05:00 | 1 h | ✗ (0.42) | ✗ (0.87) | ✗ (0.90) | ✗ (0.00) |
| 27 | drop@0 | 2023-10-09 05:00 | 2 h | ✗ (0.25) | ✗ (0.97) | ✗ (0.83) | ✗ (0.03) |
| 28 | leakage@0.15 | 2023-10-12 01:00 | 140 h | ✗ (0.67) | ✓ (1.14) | ✗ (0.94) | ✓ (1.07) |
| 29 | leakage@0.15 | 2023-10-23 15:00 | 117 h | ✗ (0.44) | ✗ (0.78) | ✗ (0.90) | ✓ (1.02) |
| 30 | spike@5 | 2023-11-05 14:00 | 1 h | ✗ (0.28) | ✓ (1.94) | ✗ (0.92) | ✗ (0.00) |
| 31 | spike@5 | 2023-11-15 02:00 | 1 h | ✗ (0.24) | ✗ (0.70) | ✗ (0.84) | ✗ (0.00) |
| 32 | drop@0 | 2023-11-18 14:00 | 2 h | ✗ (0.27) | ✗ (0.71) | ✗ (0.81) | ✗ (0.01) |

### raw / wmz_3  (Schiene: gt_stat)

| # | Event | Start | Dauer | zscore | lof | iforest | constancy |
|---|---|---|---|---|---|---|---|
| 1 | spike@3 | 2023-05-01 06:00 | 1 h | ✓ (1.16) | ✓ (1.57) | ✓ (1.04) | ✗ (0.00) |
| 2 | spike@10 | 2023-05-03 22:00 | 1 h | ✗ (0.90) | ✓ (1.66) | ✓ (1.02) | ✗ (0.00) |
| 3 | leakage@0.3 | 2023-05-06 17:00 | 119 h | ✗ (0.31) | ✓ (1.06) | ✗ (0.99) | ✓ (1.05) |
| 4 | spike@10 | 2023-05-14 05:00 | 1 h | ✗ (0.29) | ✓ (1.45) | ✓ (1.02) | ✗ (0.00) |
| 5 | spike@10 | 2023-05-19 00:00 | 1 h | ✗ (0.50) | ✓ (1.23) | ✗ (0.97) | ✗ (0.00) |
| 6 | spike@5 | 2023-05-25 07:00 | 1 h | ✗ (0.40) | ✗ (0.72) | ✗ (0.89) | ✗ (0.00) |
| 7 | drop@0.25 | 2023-05-30 08:00 | 3 h | ✗ (0.22) | ✗ (0.77) | ✗ (0.80) | ✗ (0.00) |
| 8 | leakage@0.3 | 2023-06-02 10:00 | 153 h | ✗ (0.42) | ✗ (0.85) | ✗ (0.94) | ✗ (0.95) |
| 9 | leakage@0.15 | 2023-06-13 04:00 | 143 h | ✗ (0.36) | ✗ (0.78) | ✗ (0.90) | ✗ (0.33) |
| 10 | leakage@0.3 | 2023-06-27 21:00 | 82 h | ✗ (0.29) | ✗ (0.77) | ✗ (0.87) | ✗ (0.31) |
| 11 | plateau | 2023-07-05 04:00 | 22 h | ✗ (0.32) | ✗ (0.65) | ✗ (0.86) | ✗ (0.21) |
| 12 | leakage@0.15 | 2023-07-15 00:00 | 153 h | ✗ (0.45) | ✗ (0.72) | ✗ (0.86) | ✗ (0.00) |
| 13 | drop@0 | 2023-08-09 08:00 | 3 h | ✗ (0.22) | ✗ (0.68) | ✗ (0.85) | ✗ (0.00) |
| 14 | leakage@0.15 | 2023-08-11 17:00 | 113 h | ✗ (0.33) | ✗ (0.72) | ✗ (0.87) | ✗ (0.73) |
| 15 | leakage@0.3 | 2023-08-19 13:00 | 49 h | ✗ (0.34) | ✗ (0.69) | ✗ (0.86) | ✗ (0.00) |
| 16 | drop@0.25 | 2023-08-28 12:00 | 2 h | ✗ (0.23) | ✗ (0.63) | ✗ (0.83) | ✗ (0.00) |
| 17 | leakage@0.15 | 2023-08-31 04:00 | 150 h | ✗ (0.29) | ✗ (0.79) | ✗ (0.96) | ✗ (0.87) |
| 18 | plateau | 2023-09-12 06:00 | 21 h | ✗ (0.32) | ✗ (0.70) | ✗ (0.82) | ✗ (0.29) |
| 19 | plateau | 2023-09-19 11:00 | 13 h | ✗ (0.32) | ✗ (0.70) | ✗ (0.82) | ✗ (0.19) |
| 20 | plateau | 2023-09-24 02:00 | 23 h | ✗ (0.29) | ✗ (0.73) | ✗ (0.92) | ✗ (0.49) |
| 21 | drop@0.25 | 2023-09-29 08:00 | 3 h | ✗ (0.23) | ✗ (0.69) | ✗ (0.84) | ✗ (0.00) |
| 22 | spike@10 | 2023-10-05 09:00 | 1 h | ✗ (0.96) | ✓ (1.55) | ✗ (0.98) | ✗ (0.00) |
| 23 | drop@0 | 2023-10-09 22:00 | 2 h | ✗ (0.27) | ✗ (0.84) | ✗ (0.90) | ✗ (0.69) |
| 24 | spike@3 | 2023-10-14 11:00 | 1 h | ✗ (0.29) | ✗ (0.73) | ✗ (0.95) | ✗ (0.24) |
| 25 | spike@5 | 2023-10-18 16:00 | 1 h | ✗ (0.56) | ✓ (1.15) | ✗ (0.92) | ✗ (0.00) |
| 26 | drop@0 | 2023-10-24 20:00 | 2 h | ✗ (0.27) | ✗ (0.75) | ✗ (0.88) | ✗ (0.51) |
| 27 | spike@3 | 2023-10-29 03:00 | 1 h | ✗ (0.29) | ✓ (1.24) | ✓ (1.01) | ✗ (0.11) |
| 28 | spike@5 | 2023-11-02 13:00 | 1 h | ✗ (0.76) | ✓ (1.20) | ✗ (0.92) | ✗ (0.00) |
| 29 | spike@3 | 2023-11-04 16:00 | 1 h | ✗ (0.29) | ✗ (0.99) | ✗ (0.97) | ✗ (0.19) |
| 30 | drop@0.25 | 2023-11-08 09:00 | 2 h | ✗ (0.27) | ✗ (0.75) | ✗ (0.85) | ✗ (0.00) |
| 31 | drop@0 | 2023-11-13 07:00 | 3 h | ✗ (0.27) | ✗ (0.94) | ✗ (0.88) | ✗ (0.00) |
| 32 | spike@5 | 2023-11-18 01:00 | 1 h | ✗ (0.31) | ✓ (1.58) | ✓ (1.02) | ✗ (0.00) |

### residual / wmz_1  (Schiene: gt_stat)

| # | Event | Start | Dauer | zscore | lof | iforest |
|---|---|---|---|---|---|---|
| 1 | leakage@0.15 | 2023-05-03 06:00 | 146 h | ✗ (0.90) | ✓ (1.17) | ✗ (0.96) |
| 2 | spike@5 | 2023-05-12 09:00 | 1 h | ✓ (7.02) | ✓ (11.13) | ✗ (0.97) |
| 3 | spike@5 | 2023-05-19 21:00 | 1 h | ✓ (5.31) | ✓ (8.61) | ✗ (0.97) |
| 4 | drop@0 | 2023-06-03 07:00 | 2 h | ✓ (1.42) | ✗ (0.57) | ✗ (0.99) |
| 5 | spike@3 | 2023-06-07 14:00 | 1 h | ✓ (3.21) | ✓ (5.53) | ✓ (1.04) |
| 6 | spike@10 | 2023-06-10 11:00 | 1 h | ✓ (37.92) | ✓ (61.06) | ✓ (1.02) |
| 7 | drop@0.25 | 2023-06-13 20:00 | 1 h | ✓ (1.16) | ✗ (0.60) | ✗ (0.97) |
| 8 | plateau | 2023-06-18 11:00 | 10 h | ✗ (0.39) | ✗ (0.55) | ✗ (0.89) |
| 9 | plateau | 2023-06-23 23:00 | 15 h | ✗ (0.48) | ✗ (0.62) | ✗ (0.91) |
| 10 | leakage@0.3 | 2023-06-27 23:00 | 138 h | ✗ (0.97) | ✓ (1.36) | ✗ (0.98) |
| 11 | spike@3 | 2023-07-06 07:00 | 1 h | ✓ (3.55) | ✓ (6.32) | ✗ (0.97) |
| 12 | leakage@0.15 | 2023-07-14 14:00 | 132 h | ✗ (0.97) | ✓ (1.18) | ✓ (1.03) |
| 13 | spike@10 | 2023-07-22 17:00 | 1 h | ✓ (11.99) | ✓ (19.05) | ✗ (0.99) |
| 14 | spike@5 | 2023-07-30 01:00 | 1 h | ✓ (5.35) | ✓ (8.65) | ✗ (0.97) |
| 15 | spike@5 | 2023-08-05 05:00 | 1 h | ✓ (5.03) | ✓ (8.24) | ✗ (0.97) |
| 16 | drop@0.25 | 2023-08-08 18:00 | 3 h | ✓ (1.30) | ✗ (0.61) | ✗ (0.99) |
| 17 | leakage@0.3 | 2023-08-12 05:00 | 160 h | ✓ (1.00) | ✓ (1.44) | ✓ (1.01) |
| 18 | spike@3 | 2023-08-23 12:00 | 1 h | ✓ (3.13) | ✓ (5.49) | ✓ (1.02) |
| 19 | leakage@0.3 | 2023-08-31 17:00 | 157 h | ✗ (0.96) | ✓ (1.22) | ✗ (0.96) |
| 20 | drop@0.25 | 2023-09-10 06:00 | 1 h | ✗ (0.94) | ✗ (0.65) | ✗ (0.95) |
| 21 | drop@0 | 2023-09-14 23:00 | 1 h | ✓ (1.42) | ✗ (0.58) | ✗ (0.97) |
| 22 | leakage@0.3 | 2023-09-17 23:00 | 63 h | ✗ (0.89) | ✓ (1.12) | ✓ (1.01) |
| 23 | leakage@0.15 | 2023-09-24 23:00 | 48 h | ✗ (0.75) | ✗ (0.92) | ✗ (0.95) |
| 24 | spike@10 | 2023-09-30 08:00 | 1 h | ✓ (12.67) | ✓ (20.30) | ✗ (0.96) |
| 25 | drop@0.25 | 2023-10-06 20:00 | 3 h | ✓ (1.25) | ✗ (0.65) | ✗ (0.99) |
| 26 | spike@10 | 2023-10-10 10:00 | 1 h | ✓ (12.15) | ✓ (19.40) | ✗ (0.97) |
| 27 | leakage@0.15 | 2023-10-17 11:00 | 93 h | ✗ (0.86) | ✓ (1.16) | ✗ (0.93) |
| 28 | plateau | 2023-10-24 01:00 | 11 h | ✗ (0.63) | ✗ (0.75) | ✗ (0.91) |
| 29 | drop@0 | 2023-10-28 23:00 | 2 h | ✓ (1.50) | ✗ (0.58) | ✗ (0.98) |
| 30 | drop@0 | 2023-11-01 11:00 | 2 h | ✓ (1.83) | ✗ (0.64) | ✗ (0.98) |
| 31 | spike@3 | 2023-11-08 10:00 | 1 h | ✓ (2.95) | ✓ (5.28) | ✗ (0.97) |
| 32 | plateau | 2023-11-15 06:00 | 16 h | ✗ (0.30) | ✗ (0.39) | ✗ (0.80) |

### residual / wmz_2  (Schiene: gt_stat)

| # | Event | Start | Dauer | zscore | lof | iforest |
|---|---|---|---|---|---|---|
| 1 | spike@10 | 2023-05-05 17:00 | 1 h | ✓ (1.11) | ✗ (0.84) | ✗ (0.89) |
| 2 | spike@10 | 2023-05-09 13:00 | 1 h | ✓ (4.92) | ✓ (5.02) | ✗ (0.98) |
| 3 | spike@3 | 2023-05-13 23:00 | 1 h | ✗ (0.22) | ✗ (0.64) | ✗ (0.71) |
| 4 | spike@5 | 2023-05-18 01:00 | 1 h | ✗ (0.96) | ✓ (1.92) | ✗ (0.91) |
| 5 | spike@10 | 2023-05-25 09:00 | 1 h | ✓ (1.60) | ✓ (1.05) | ✗ (0.92) |
| 6 | plateau | 2023-05-28 19:00 | 21 h | ✗ (0.96) | ✗ (0.84) | ✗ (0.91) |
| 7 | leakage@0.15 | 2023-06-06 11:00 | 65 h | ✗ (0.38) | ✓ (1.04) | ✗ (0.77) |
| 8 | plateau | 2023-06-13 22:00 | 15 h | ✗ (0.29) | ✗ (0.94) | ✗ (0.76) |
| 9 | drop@0 | 2023-06-22 10:00 | 2 h | ✗ (0.28) | ✓ (1.01) | ✗ (0.75) |
| 10 | leakage@0.3 | 2023-06-28 23:00 | 111 h | ✗ (0.29) | ✗ (0.87) | ✗ (0.71) |
| 11 | leakage@0.3 | 2023-07-05 20:00 | 75 h | ✗ (0.42) | ✓ (1.19) | ✗ (0.78) |
| 12 | plateau | 2023-07-14 03:00 | 9 h | ✗ (0.24) | ✗ (0.88) | ✗ (0.68) |
| 13 | drop@0 | 2023-07-17 07:00 | 2 h | ✗ (0.23) | ✗ (0.80) | ✗ (0.72) |
| 14 | spike@3 | 2023-07-22 19:00 | 1 h | ✗ (0.23) | ✗ (0.79) | ✗ (0.72) |
| 15 | leakage@0.3 | 2023-07-25 18:00 | 54 h | ✗ (0.39) | ✗ (0.85) | ✗ (0.75) |
| 16 | drop@0.25 | 2023-07-31 07:00 | 3 h | ✗ (0.18) | ✗ (0.67) | ✗ (0.67) |
| 17 | spike@10 | 2023-08-09 12:00 | 1 h | ✓ (3.11) | ✓ (2.65) | ✗ (0.95) |
| 18 | drop@0.25 | 2023-08-14 23:00 | 3 h | ✗ (0.22) | ✗ (0.66) | ✗ (0.68) |
| 19 | leakage@0.3 | 2023-08-17 16:00 | 66 h | ✗ (0.42) | ✗ (0.93) | ✗ (0.76) |
| 20 | spike@3 | 2023-08-24 14:00 | 1 h | ✗ (0.72) | ✓ (1.35) | ✗ (0.87) |
| 21 | spike@5 | 2023-08-28 22:00 | 1 h | ✗ (0.44) | ✗ (0.78) | ✗ (0.74) |
| 22 | drop@0.25 | 2023-09-07 08:00 | 2 h | ✗ (0.23) | ✗ (0.69) | ✗ (0.71) |
| 23 | leakage@0.15 | 2023-09-11 11:00 | 73 h | ✗ (0.42) | ✗ (0.99) | ✗ (0.76) |
| 24 | plateau | 2023-09-21 00:00 | 16 h | ✗ (0.41) | ✓ (1.09) | ✗ (0.83) |
| 25 | drop@0.25 | 2023-09-27 08:00 | 3 h | ✗ (0.43) | ✗ (0.74) | ✗ (0.75) |
| 26 | spike@3 | 2023-10-05 05:00 | 1 h | ✗ (0.65) | ✗ (0.70) | ✗ (0.79) |
| 27 | drop@0 | 2023-10-09 05:00 | 2 h | ✗ (0.68) | ✗ (0.85) | ✗ (0.77) |
| 28 | leakage@0.15 | 2023-10-12 01:00 | 140 h | ✗ (0.72) | ✗ (0.94) | ✗ (0.79) |
| 29 | leakage@0.15 | 2023-10-23 15:00 | 117 h | ✗ (0.46) | ✗ (0.97) | ✗ (0.76) |
| 30 | spike@5 | 2023-11-05 14:00 | 1 h | ✗ (0.47) | ✗ (0.76) | ✗ (0.76) |
| 31 | spike@5 | 2023-11-15 02:00 | 1 h | ✗ (0.47) | ✗ (0.69) | ✗ (0.74) |
| 32 | drop@0 | 2023-11-18 14:00 | 2 h | ✗ (0.17) | ✗ (0.59) | ✗ (0.67) |

### residual / wmz_3  (Schiene: gt_stat)

| # | Event | Start | Dauer | zscore | lof | iforest |
|---|---|---|---|---|---|---|
| 1 | spike@3 | 2023-05-01 06:00 | 1 h | ✓ (2.30) | ✓ (1.61) | ✓ (1.17) |
| 2 | spike@10 | 2023-05-03 22:00 | 1 h | ✓ (1.65) | ✓ (1.22) | ✓ (1.06) |
| 3 | leakage@0.3 | 2023-05-06 17:00 | 119 h | ✗ (0.66) | ✗ (0.94) | ✗ (0.92) |
| 4 | spike@10 | 2023-05-14 05:00 | 1 h | ✗ (0.42) | ✗ (0.45) | ✗ (0.86) |
| 5 | spike@10 | 2023-05-19 00:00 | 1 h | ✗ (0.91) | ✗ (0.42) | ✗ (0.98) |
| 6 | spike@5 | 2023-05-25 07:00 | 1 h | ✗ (0.76) | ✗ (0.54) | ✗ (0.94) |
| 7 | drop@0.25 | 2023-05-30 08:00 | 3 h | ✗ (0.35) | ✗ (0.66) | ✗ (0.80) |
| 8 | leakage@0.3 | 2023-06-02 10:00 | 153 h | ✗ (0.43) | ✓ (1.05) | ✗ (0.79) |
| 9 | leakage@0.15 | 2023-06-13 04:00 | 143 h | ✗ (0.39) | ✗ (0.55) | ✗ (0.74) |
| 10 | leakage@0.3 | 2023-06-27 21:00 | 82 h | ✗ (0.39) | ✗ (0.59) | ✗ (0.71) |
| 11 | plateau | 2023-07-05 04:00 | 22 h | ✗ (0.39) | ✗ (0.47) | ✗ (0.68) |
| 12 | leakage@0.15 | 2023-07-15 00:00 | 153 h | ✗ (0.51) | ✗ (0.83) | ✗ (0.82) |
| 13 | drop@0 | 2023-08-09 08:00 | 3 h | ✗ (0.26) | ✗ (0.46) | ✗ (0.74) |
| 14 | leakage@0.15 | 2023-08-11 17:00 | 113 h | ✗ (0.46) | ✗ (0.89) | ✗ (0.75) |
| 15 | leakage@0.3 | 2023-08-19 13:00 | 49 h | ✗ (0.46) | ✗ (0.53) | ✗ (0.74) |
| 16 | drop@0.25 | 2023-08-28 12:00 | 2 h | ✗ (0.28) | ✗ (0.45) | ✗ (0.67) |
| 17 | leakage@0.15 | 2023-08-31 04:00 | 150 h | ✗ (0.42) | ✗ (0.82) | ✗ (0.81) |
| 18 | plateau | 2023-09-12 06:00 | 21 h | ✗ (0.46) | ✗ (0.62) | ✗ (0.76) |
| 19 | plateau | 2023-09-19 11:00 | 13 h | ✗ (0.37) | ✗ (0.47) | ✗ (0.68) |
| 20 | plateau | 2023-09-24 02:00 | 23 h | ✗ (0.33) | ✗ (0.64) | ✗ (0.75) |
| 21 | drop@0.25 | 2023-09-29 08:00 | 3 h | ✗ (0.34) | ✗ (0.52) | ✗ (0.77) |
| 22 | spike@10 | 2023-10-05 09:00 | 1 h | ✓ (1.67) | ✓ (1.21) | ✓ (1.06) |
| 23 | drop@0 | 2023-10-09 22:00 | 2 h | ✗ (0.27) | ✗ (0.40) | ✗ (0.79) |
| 24 | spike@3 | 2023-10-14 11:00 | 1 h | ✗ (0.19) | ✗ (0.42) | ✗ (0.79) |
| 25 | spike@5 | 2023-10-18 16:00 | 1 h | ✓ (1.14) | ✗ (0.61) | ✓ (1.03) |
| 26 | drop@0 | 2023-10-24 20:00 | 2 h | ✗ (0.20) | ✗ (0.42) | ✗ (0.80) |
| 27 | spike@3 | 2023-10-29 03:00 | 1 h | ✗ (0.32) | ✗ (0.41) | ✗ (0.84) |
| 28 | spike@5 | 2023-11-02 13:00 | 1 h | ✓ (1.08) | ✗ (0.51) | ✓ (1.02) |
| 29 | spike@3 | 2023-11-04 16:00 | 1 h | ✗ (0.26) | ✗ (0.42) | ✗ (0.84) |
| 30 | drop@0.25 | 2023-11-08 09:00 | 2 h | ✗ (0.45) | ✗ (0.46) | ✗ (0.86) |
| 31 | drop@0 | 2023-11-13 07:00 | 3 h | ✓ (1.02) | ✗ (0.97) | ✓ (1.02) |
| 32 | spike@5 | 2023-11-18 01:00 | 1 h | ✓ (1.01) | ✗ (0.46) | ✓ (1.00) |

### trend / wmz_1  (Schiene: gt_nonstat)

| # | Event | Start | Dauer | pelt |
|---|---|---|---|---|
| 1 | drift@0.01 | 2023-05-03 01:00 | 586 h | ✓ (2.00) |
| 2 | drift@0.005 | 2023-05-30 13:00 | 699 h | ✓ (2.00) |
| 3 | structural_break@0.2 | 2023-06-30 21:00 | 1 h | ✗ (0.00) |
| 4 | structural_break@0.1 | 2023-07-04 04:00 | 1 h | ✗ (0.00) |
| 5 | drift@0.01 | 2023-07-06 06:00 | 392 h | ✓ (2.00) |
| 6 | structural_break@0.1 | 2023-07-27 05:00 | 1 h | ✗ (0.00) |
| 7 | drift@0.005 | 2023-08-01 18:00 | 362 h | ✗ (0.00) |
| 8 | drift@0.005 | 2023-08-22 06:00 | 536 h | ✗ (0.00) |
| 9 | drift@0.01 | 2023-09-16 21:00 | 372 h | ✓ (2.00) |
| 10 | drift@0.005 | 2023-10-04 17:00 | 599 h | ✓ (2.00) |
| 11 | structural_break@0.1 | 2023-11-02 16:00 | 1 h | ✗ (0.00) |
| 12 | structural_break@0.2 | 2023-11-05 22:00 | 1 h | ✓ (2.00) |
| 13 | structural_break@0.2 | 2023-11-08 10:00 | 1 h | ✗ (0.00) |
| 14 | structural_break@0.2 | 2023-11-11 23:00 | 1 h | ✗ (0.00) |
| 15 | structural_break@0.1 | 2023-11-15 15:00 | 1 h | ✗ (0.00) |
| 16 | drift@0.01 | 2023-11-18 20:00 | 4 h | ✗ (0.00) |

### trend / wmz_2  (Schiene: gt_nonstat)

| # | Event | Start | Dauer | pelt |
|---|---|---|---|---|
| 1 | drift@0.01 | 2023-05-01 04:00 | 450 h | ✗ (0.00) |
| 2 | drift@0.01 | 2023-05-22 02:00 | 624 h | ✓ (2.00) |
| 3 | drift@0.01 | 2023-06-20 08:00 | 357 h | ✗ (0.00) |
| 4 | drift@0.005 | 2023-07-09 15:00 | 347 h | ✗ (0.00) |
| 5 | structural_break@0.1 | 2023-07-28 08:00 | 1 h | ✗ (0.00) |
| 6 | structural_break@0.2 | 2023-07-30 13:00 | 1 h | ✗ (0.00) |
| 7 | drift@0.005 | 2023-08-01 17:00 | 472 h | ✗ (0.00) |
| 8 | structural_break@0.1 | 2023-08-23 21:00 | 1 h | ✗ (0.00) |
| 9 | structural_break@0.1 | 2023-08-27 04:00 | 1 h | ✗ (0.00) |
| 10 | structural_break@0.2 | 2023-08-29 08:00 | 1 h | ✗ (0.00) |
| 11 | structural_break@0.1 | 2023-09-05 00:00 | 1 h | ✗ (0.00) |
| 12 | structural_break@0.2 | 2023-09-10 03:00 | 1 h | ✗ (0.00) |
| 13 | structural_break@0.2 | 2023-09-13 02:00 | 1 h | ✗ (0.00) |
| 14 | drift@0.005 | 2023-09-15 10:00 | 717 h | ✓ (2.00) |
| 15 | drift@0.01 | 2023-10-17 19:00 | 521 h | ✗ (0.00) |
| 16 | drift@0.005 | 2023-11-10 19:00 | 197 h | ✗ (0.00) |

### trend / wmz_3  (Schiene: gt_nonstat)

| # | Event | Start | Dauer | pelt |
|---|---|---|---|---|
| 1 | structural_break@0.1 | 2023-05-01 12:00 | 1 h | ✗ (0.00) |
| 2 | drift@0.01 | 2023-05-05 06:00 | 438 h | ✓ (2.00) |
| 3 | structural_break@0.2 | 2023-05-25 13:00 | 1 h | ✗ (0.00) |
| 4 | drift@0.005 | 2023-06-03 11:00 | 660 h | ✗ (0.00) |
| 5 | drift@0.01 | 2023-07-06 04:00 | 545 h | ✗ (0.00) |
| 6 | drift@0.005 | 2023-07-30 23:00 | 613 h | ✗ (0.00) |
| 7 | structural_break@0.2 | 2023-08-29 03:00 | 1 h | ✗ (0.00) |
| 8 | structural_break@0.2 | 2023-08-31 12:00 | 1 h | ✗ (0.00) |
| 9 | structural_break@0.1 | 2023-09-05 10:00 | 1 h | ✗ (0.00) |
| 10 | structural_break@0.1 | 2023-09-07 17:00 | 1 h | ✗ (0.00) |
| 11 | drift@0.005 | 2023-09-10 19:00 | 337 h | ✗ (0.00) |
| 12 | drift@0.005 | 2023-10-01 13:00 | 399 h | ✓ (2.00) |
| 13 | structural_break@0.2 | 2023-10-22 13:00 | 1 h | ✗ (0.00) |
| 14 | structural_break@0.1 | 2023-10-27 00:00 | 1 h | ✗ (0.00) |
| 15 | drift@0.01 | 2023-10-30 05:00 | 394 h | ✓ (2.00) |
| 16 | drift@0.01 | 2023-11-18 01:00 | 23 h | ✗ (0.00) |

---

## C. False-Positive-Kandidaten (Alarme außerhalb jedes GT-Events)

Die stärksten Alarm-Segmente, die **kein** injiziertes Event treffen. Zentrale Frage der qualitativen Analyse: echter Fehlalarm oder reale (Makro-)Anomalie? Mit dem Plot-Befehl das Fenster inspizieren und gegen `monatlich_makro_events.png` (COVID/EnSikuMaV) abgleichen.

### raw / wmz_1 / zscore

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-06-10 12:00 | 2023-06-11 10:00 | 23 h | 39.529 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model zscore --start 2023-06-07 --end 2023-06-14` |
| 2023-09-30 09:00 | 2023-10-01 07:00 | 23 h | 11.765 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model zscore --start 2023-09-27 --end 2023-10-04` |
| 2023-07-22 18:00 | 2023-07-23 16:00 | 23 h | 11.145 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model zscore --start 2023-07-19 --end 2023-07-26` |
| 2023-10-10 11:00 | 2023-10-11 09:00 | 23 h | 11.019 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model zscore --start 2023-10-07 --end 2023-10-14` |
| 2023-05-12 10:00 | 2023-05-13 08:00 | 23 h | 6.656 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model zscore --start 2023-05-09 --end 2023-05-16` |

### raw / wmz_1 / lof

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-06-10 12:00 | 2023-06-11 10:00 | 23 h | 44.362 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model lof --start 2023-06-07 --end 2023-06-14` |
| 2023-09-30 09:00 | 2023-10-01 07:00 | 23 h | 11.367 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model lof --start 2023-09-27 --end 2023-10-04` |
| 2023-10-10 11:00 | 2023-10-11 09:00 | 23 h | 10.786 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model lof --start 2023-10-07 --end 2023-10-14` |
| 2023-07-22 18:00 | 2023-07-23 16:00 | 23 h | 10.648 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model lof --start 2023-07-19 --end 2023-07-26` |
| 2023-05-12 10:00 | 2023-05-13 08:00 | 23 h | 5.599 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model lof --start 2023-05-09 --end 2023-05-16` |

### raw / wmz_1 / iforest

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-06-10 12:00 | 2023-06-11 10:00 | 23 h | 0.665 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model iforest --start 2023-06-07 --end 2023-06-14` |
| 2023-07-22 18:00 | 2023-07-22 23:00 | 6 h | 0.649 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model iforest --start 2023-07-19 --end 2023-07-25` |
| 2023-06-10 07:00 | 2023-06-10 08:00 | 2 h | 0.648 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model iforest --start 2023-06-07 --end 2023-06-13` |
| 2023-09-30 09:00 | 2023-09-30 14:00 | 6 h | 0.641 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model iforest --start 2023-09-27 --end 2023-10-03` |
| 2023-07-23 09:00 | 2023-07-23 16:00 | 8 h | 0.633 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model iforest --start 2023-07-20 --end 2023-07-26` |

### raw / wmz_1 / constancy

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-10-31 02:00 | 2023-10-31 02:00 | 1 h | 0.972 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model constancy --start 2023-10-28 --end 2023-11-03` |
| 2023-05-27 01:00 | 2023-05-27 02:00 | 2 h | 0.972 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model constancy --start 2023-05-24 --end 2023-05-30` |
| 2023-08-06 20:00 | 2023-08-06 23:00 | 4 h | 0.963 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model constancy --start 2023-08-03 --end 2023-08-09` |
| 2023-11-09 03:00 | 2023-11-09 04:00 | 2 h | 0.958 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model constancy --start 2023-11-06 --end 2023-11-12` |
| 2023-05-02 04:00 | 2023-05-02 04:00 | 1 h | 0.957 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_1 --model constancy --start 2023-04-29 --end 2023-05-05` |

### raw / wmz_2 / zscore

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-09 14:00 | 2023-05-09 18:00 | 5 h | 14.158 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model zscore --start 2023-05-06 --end 2023-05-12` |
| 2023-08-09 13:00 | 2023-08-09 17:00 | 5 h | 8.101 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model zscore --start 2023-08-06 --end 2023-08-12` |

### raw / wmz_2 / lof

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-09 14:00 | 2023-05-10 12:00 | 23 h | 4.847 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model lof --start 2023-05-06 --end 2023-05-13` |
| 2023-08-09 13:00 | 2023-08-10 11:00 | 23 h | 3.903 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model lof --start 2023-08-06 --end 2023-08-13` |
| 2023-06-10 11:00 | 2023-06-10 16:00 | 6 h | 3.451 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model lof --start 2023-06-07 --end 2023-06-13` |
| 2023-11-15 22:00 | 2023-11-16 00:00 | 3 h | 2.894 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model lof --start 2023-11-12 --end 2023-11-19` |
| 2023-10-20 22:00 | 2023-10-20 23:00 | 2 h | 2.302 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model lof --start 2023-10-17 --end 2023-10-23` |

### raw / wmz_2 / constancy

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-08-16 04:00 | 2023-08-16 04:00 | 1 h | 0.991 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model constancy --start 2023-08-13 --end 2023-08-19` |
| 2023-05-22 22:00 | 2023-05-23 02:00 | 5 h | 0.989 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model constancy --start 2023-05-19 --end 2023-05-26` |
| 2023-10-01 12:00 | 2023-10-01 15:00 | 4 h | 0.985 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model constancy --start 2023-09-28 --end 2023-10-04` |
| 2023-07-30 07:00 | 2023-07-30 11:00 | 5 h | 0.982 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model constancy --start 2023-07-27 --end 2023-08-02` |
| 2023-06-22 02:00 | 2023-06-22 02:00 | 1 h | 0.981 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model constancy --start 2023-06-19 --end 2023-06-25` |

### raw / wmz_3 / lof

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-03 23:00 | 2023-05-04 03:00 | 5 h | 2.465 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model lof --start 2023-04-30 --end 2023-05-07` |
| 2023-11-18 02:00 | 2023-11-18 06:00 | 5 h | 2.381 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model lof --start 2023-11-15 --end 2023-11-21` |
| 2023-10-05 10:00 | 2023-10-05 14:00 | 5 h | 2.143 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model lof --start 2023-10-02 --end 2023-10-08` |
| 2023-05-02 00:00 | 2023-05-02 05:00 | 6 h | 2.066 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model lof --start 2023-04-29 --end 2023-05-05` |
| 2023-05-14 06:00 | 2023-05-14 10:00 | 5 h | 2.061 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model lof --start 2023-05-11 --end 2023-05-17` |

### raw / wmz_3 / iforest

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-11-18 03:00 | 2023-11-18 06:00 | 4 h | 0.613 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --start 2023-11-15 --end 2023-11-21` |
| 2023-05-06 05:00 | 2023-05-06 05:00 | 1 h | 0.607 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --start 2023-05-03 --end 2023-05-09` |
| 2023-10-29 00:00 | 2023-10-29 00:00 | 1 h | 0.604 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --start 2023-10-26 --end 2023-11-01` |
| 2023-10-29 05:00 | 2023-10-29 05:00 | 1 h | 0.602 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --start 2023-10-26 --end 2023-11-01` |
| 2023-05-01 10:00 | 2023-05-01 11:00 | 2 h | 0.601 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --start 2023-04-28 --end 2023-05-04` |

### raw / wmz_3 / constancy

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-05 17:00 | 2023-05-05 17:00 | 1 h | 0.998 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model constancy --start 2023-05-02 --end 2023-05-08` |
| 2023-10-06 19:00 | 2023-10-07 20:00 | 26 h | 0.998 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model constancy --start 2023-10-03 --end 2023-10-10` |
| 2023-10-22 18:00 | 2023-10-22 22:00 | 5 h | 0.995 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model constancy --start 2023-10-19 --end 2023-10-25` |
| 2023-05-11 20:00 | 2023-05-12 03:00 | 8 h | 0.988 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model constancy --start 2023-05-08 --end 2023-05-15` |
| 2023-10-30 00:00 | 2023-10-30 01:00 | 2 h | 0.984 | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model constancy --start 2023-10-27 --end 2023-11-02` |

### residual / wmz_1 / zscore

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-06-10 12:00 | 2023-06-11 10:00 | 23 h | 47.627 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model zscore --start 2023-06-07 --end 2023-06-14` |
| 2023-09-30 09:00 | 2023-10-01 07:00 | 23 h | 14.334 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model zscore --start 2023-09-27 --end 2023-10-04` |
| 2023-07-22 18:00 | 2023-07-23 16:00 | 23 h | 13.497 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model zscore --start 2023-07-19 --end 2023-07-26` |
| 2023-10-10 11:00 | 2023-10-11 09:00 | 23 h | 13.452 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model zscore --start 2023-10-07 --end 2023-10-14` |
| 2023-05-12 10:00 | 2023-05-13 08:00 | 23 h | 8.154 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model zscore --start 2023-05-09 --end 2023-05-16` |

### residual / wmz_1 / lof

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-06-10 12:00 | 2023-06-11 10:00 | 23 h | 29.498 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --start 2023-06-07 --end 2023-06-14` |
| 2023-09-30 09:00 | 2023-10-01 07:00 | 23 h | 8.257 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --start 2023-09-27 --end 2023-10-04` |
| 2023-10-10 11:00 | 2023-10-11 09:00 | 23 h | 7.829 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --start 2023-10-07 --end 2023-10-14` |
| 2023-07-22 18:00 | 2023-07-23 16:00 | 23 h | 7.685 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --start 2023-07-19 --end 2023-07-26` |
| 2023-05-12 10:00 | 2023-05-13 08:00 | 23 h | 4.321 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --start 2023-05-09 --end 2023-05-16` |

### residual / wmz_1 / iforest

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-29 14:00 | 2023-05-29 14:00 | 1 h | 0.698 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model iforest --start 2023-05-26 --end 2023-06-01` |
| 2023-05-18 06:00 | 2023-05-18 06:00 | 1 h | 0.693 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model iforest --start 2023-05-15 --end 2023-05-21` |
| 2023-05-01 14:00 | 2023-05-01 15:00 | 2 h | 0.691 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model iforest --start 2023-04-28 --end 2023-05-04` |
| 2023-05-01 17:00 | 2023-05-01 18:00 | 2 h | 0.691 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model iforest --start 2023-04-28 --end 2023-05-04` |
| 2023-06-10 17:00 | 2023-06-10 18:00 | 2 h | 0.689 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model iforest --start 2023-06-07 --end 2023-06-13` |

### residual / wmz_2 / zscore

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-09 14:00 | 2023-05-10 12:00 | 23 h | 12.438 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model zscore --start 2023-05-06 --end 2023-05-13` |
| 2023-08-09 13:00 | 2023-08-10 11:00 | 23 h | 7.214 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model zscore --start 2023-08-06 --end 2023-08-13` |

### residual / wmz_2 / lof

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-09 14:00 | 2023-05-10 12:00 | 23 h | 5.006 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model lof --start 2023-05-06 --end 2023-05-13` |
| 2023-08-09 13:00 | 2023-08-10 11:00 | 23 h | 2.444 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model lof --start 2023-08-06 --end 2023-08-13` |
| 2023-06-10 11:00 | 2023-06-10 17:00 | 7 h | 2.211 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model lof --start 2023-06-07 --end 2023-06-13` |
| 2023-10-03 12:00 | 2023-10-03 14:00 | 3 h | 2.139 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model lof --start 2023-09-30 --end 2023-10-06` |
| 2023-08-15 14:00 | 2023-08-15 14:00 | 1 h | 2.026 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_2 --model lof --start 2023-08-12 --end 2023-08-18` |

### residual / wmz_3 / zscore

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-01 07:00 | 2023-05-01 23:00 | 17 h | 2.481 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model zscore --start 2023-04-28 --end 2023-05-04` |
| 2023-10-03 14:00 | 2023-10-03 16:00 | 3 h | 1.979 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model zscore --start 2023-09-30 --end 2023-10-06` |

### residual / wmz_3 / lof

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-02 00:00 | 2023-05-02 05:00 | 6 h | 5.884 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model lof --start 2023-04-29 --end 2023-05-05` |
| 2023-05-04 09:00 | 2023-05-04 09:00 | 1 h | 4.279 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model lof --start 2023-05-01 --end 2023-05-07` |
| 2023-11-14 03:00 | 2023-11-14 03:00 | 1 h | 4.159 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model lof --start 2023-11-11 --end 2023-11-17` |
| 2023-10-06 07:00 | 2023-10-06 08:00 | 2 h | 4.117 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model lof --start 2023-10-03 --end 2023-10-09` |
| 2023-10-05 10:00 | 2023-10-05 14:00 | 5 h | 3.540 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model lof --start 2023-10-02 --end 2023-10-08` |

### residual / wmz_3 / iforest

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-01 07:00 | 2023-05-01 23:00 | 17 h | 0.683 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model iforest --start 2023-04-28 --end 2023-05-04` |
| 2023-11-18 10:00 | 2023-11-18 10:00 | 1 h | 0.618 | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_3 --model iforest --start 2023-11-15 --end 2023-11-21` |

### trend / wmz_1 / pelt

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-07-01 00:00 | 2023-07-01 00:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --start 2023-06-28 --end 2023-07-04` |
| 2023-07-27 11:00 | 2023-07-27 11:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --start 2023-07-24 --end 2023-07-30` |
| 2023-08-17 12:00 | 2023-08-17 12:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --start 2023-08-14 --end 2023-08-20` |

### trend / wmz_2 / pelt

| Start | Ende | Dauer | Spitzen-Score | Plot-Befehl |
|---|---|---|---|---|
| 2023-05-19 22:00 | 2023-05-19 22:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_2 --model pelt --start 2023-05-16 --end 2023-05-22` |
| 2023-07-30 09:00 | 2023-07-30 09:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_2 --model pelt --start 2023-07-27 --end 2023-08-02` |
| 2023-09-12 03:00 | 2023-09-12 03:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_2 --model pelt --start 2023-09-09 --end 2023-09-15` |
| 2023-11-09 06:00 | 2023-11-09 06:00 | 1 h | 1.000 | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_2 --model pelt --start 2023-11-06 --end 2023-11-12` |

---

## D. Empfohlene Fälle für die schriftliche Diskussion

Kuratierte Auswahl: je Anomalietyp ein klarer Treffer (Showcase) und ein lehrreicher Fehlschlag. Spalten **Beobachtung** und **Erklärung / Theorie-Bezug** sind die fachliche Deutung; bereits ausgefüllte Zellen bleiben beim Neugenerieren erhalten (Merge über die Fall-ID).

| Fall | Variante/WMZ/Detektor | Typ | Event-Start | Urteil | Plot-Befehl | Beobachtung | Erklärung / Theorie-Bezug |
|---|---|---|---|---|---|---|---|
| F01 (Showcase-TP) | residual/wmz_1/lof | spike@10 | 2023-06-10 11:00 | TP | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --event-start '2023-06-10 11:00'` | Scharfer Spike auf ~280 kW im Residuum; LOF-Score springt auf ~28, weit über Schwelle 3.06. | Hochamplitudige Punktanomalie → großer lokaler Dichteabfall; LOF trennt sie klar vom Grundrauschen. Lehrbuchfall distanzbasierter Detektion auf dem stabilen Grundlast-Zähler wmz_1. |
| F02 (Lehr-FN) | raw/wmz_2/zscore | spike@5 | 2023-05-18 01:00 | FN | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model zscore --event-start '2023-05-18 01:00'` | Der injizierte Spike erreicht nur Score ≈ Schwelle (ratio 1.00) und ist optisch nicht von den zahlreichen natürlichen Warmwasser-Spitzen (bis 25 kW) zu trennen. | wmz_2 (Warmwasser) ist hochvolatil; ein moderater Spike hebt sich nicht ab. Eine Schwelle, die ihn fängt, würde auf die natürlichen Lastspitzen feuern → Präzisionskollaps. Direkter Beleg der WMZ-Heterogenität und des Recall/Precision-Tradeoffs. |
| F03 (Showcase-TP) | residual/wmz_1/zscore | drop@0 | 2023-11-01 11:00 | TP | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model zscore --event-start '2023-11-01 11:00'` | Abfall auf 0 kW erzeugt einen scharfen negativen Residuum-Ausschlag; z-Score springt auf ~5 (> 3.44). | Ein vollständiger Drop ist eine große Abweichung vom gleitenden Mittel → z-Score erfasst ihn zuverlässig. Auf dem nie abgeschalteten Grundlast-Zähler ist „0 kW" stark anomal. |
| F04 (Lehr-FN) | residual/wmz_1/iforest | drop@0 | 2023-06-03 07:00 | FN | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model iforest --event-start '2023-06-03 07:00'` | Gleicher Drop wie F03, aber der IForest-Score schwankt verrauscht um die Schwelle 0.65 (ratio 0.94, knapp verpasst). | IForest isoliert über den multivariaten Featurevektor (inkl. Rolling-Stats); der kurze Drop wird darin nicht ausreichend isoliert und geht im verrauschten Score unter. Zeigt IForests Schwäche auf dem Residuum (vgl. Teil A) — derselbe Drop, anderes Modell, anderes Ergebnis. |
| F05 (Showcase-TP) | raw/wmz_2/constancy | plateau | 2023-09-21 00:00 | TP | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model constancy --event-start '2023-09-21 00:00'` | Während des Plateaus bricht die Signalvarianz ein (Flachlinie); der Constancy-Score steigt auf ~1.0 über die Schwelle. | Constancy misst Varianzeinbruch gegen die eigene Historie statt Punktabweichung — genau die Signatur eines Plateaus. Der Detektor wurde **gezielt als Reaktion auf diese Plateau-Lücke entwickelt**, die alle punkt-/abweichungsbasierten Detektoren lassen (Kontrast zu F06). |
| F06 (Lehr-FN) | raw/wmz_2/zscore | plateau | 2023-05-28 19:00 | FN | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_2 --model zscore --event-start '2023-05-28 19:00'` | Das Plateau liegt als Flachlinie auf normalem Niveau (~4 kW); der z-Score bleibt flach bei ~1.5, deutlich unter Schwelle. | Jeder Plateau-Punkt liegt nahe am lokalen Mittel → Abweichung ≈ 0 → niedriger z-Score. Punktbasierte Detektoren sind blind für kollektive Konstanz-Anomalien; das motiviert den Constancy-Detektor (direkter Kontrast zu F05). |
| F07 (Showcase-TP) | residual/wmz_1/lof | leakage@0.3 | 2023-08-12 05:00 | TP | `python src/tools/plot_qualitative_case.py --variant residual --wmz wmz_1 --model lof --event-start '2023-08-12 05:00'` | Anhaltend erhöhtes Residuum über das Leakage-Fenster; LOF-Score überschreitet die Schwelle. | Die dauerhafte Niveauverschiebung bildet eine lokal abweichende Dichteregion; LOF erkennt die kollektiv verschobenen Punkte. Leakage als kontextuell/kollektive Anomalie ist auf dem saisonbereinigten Residuum gut sichtbar (Kontrast zu F08). |
| F08 (Lehr-FN) | raw/wmz_3/iforest | leakage@0.3 | 2023-05-06 17:00 | FN | `python src/tools/plot_qualitative_case.py --variant raw --wmz wmz_3 --model iforest --event-start '2023-05-06 17:00'` | Auf dem stark schwankenden Heizzähler wmz_3 bleibt der IForest-Score während der Leakage im Rauschen unter der Schwelle. | Eine moderate, langsame Niveauanhebung (0.3) verschiebt einzelne Punkte nicht aus der ohnehin breiten Verteilung von wmz_3; ohne explizites Baseline-/Kontextmerkmal verpasst IForest die schleichende Leakage. Heterogenität: dasselbe Leakage wäre auf dem ruhigen wmz_1 sichtbarer. |
| F09 (Showcase-TP) | trend/wmz_1/pelt | drift@0.01 | 2023-05-03 01:00 | TP | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --event-start '2023-05-03 01:00'` | Der Trend rampt glatt von 10.5 auf 12.4 kW; PELT setzt mehrere Change-Points entlang der Rampe (Treppung), der erste am Drift-Onset. | PELT ist für abrupte Mean-Shifts gebaut; eine Rampe approximiert es durch stückweise-konstante Segmente. Recall wird erreicht, der Drift aber in Stufen „zerschnitten" statt als ein Verlauf erkannt (Intensitäts-Gegenstück zu F10). |
| F10 (Lehr-FN) | trend/wmz_1/pelt | drift@0.005 | 2023-08-01 18:00 | FN | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --event-start '2023-08-01 18:00'` | Bei halber Intensität nur eine sanfte 0.2-kW-Welle über zwei Wochen; PELT setzt keinen Change-Point im Band (Score flach 0), feuert erst danach. | Der schwache, allmähliche Drift senkt die Segmentierungskosten nicht genug, um den Penalty zu überwinden → unter PELTs Sensitivität. Intensitäts-Gegenstück zu F09: Detektion ist eine Frage von Amplitude vs. Penalty. |
| F11 (Showcase-TP) | trend/wmz_1/pelt | structural_break@0.2 | 2023-11-05 22:00 | TP | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --event-start '2023-11-05 22:00'` | Scharfer Niveausprung (12.35 → 12.56 kW); PELT setzt den Change-Point exakt im GT-Onset-Fenster → TP. | Der abrupte Mean-Shift ist genau die Anomalieform, für die PELT konstruiert ist (l2-Kosten = Mittelwertänderung). Idealfall der Change-Point-Detektion (vgl. F12, gleicher Typ/Intensität, aber FN). |
| F12 (Lehr-FN) | trend/wmz_1/pelt | structural_break@0.2 | 2023-06-30 21:00 | FN | `python src/tools/plot_qualitative_case.py --variant trend --wmz wmz_1 --model pelt --event-start '2023-06-30 21:00'` | Ebenfalls klarer Sprung, PELT feuert sichtbar — aber der Change-Point landet knapp außerhalb des schmalen GT-Onset-Fensters → Point-Adjust wertet FN. | Die niedrige structural_break-Recall (Teil A) ist großteils ein Lokalisierungs-/Fenster-Artefakt, kein Detektionsversagen (vgl. F11, gleicher Typ/Intensität, TP). [prüfen] Toleranzfenster des Point-Adjust für Change-Points verbreitern und Recall neu bewerten. |
