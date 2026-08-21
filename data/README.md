# data/

Ein Unterordner je Datensatz, `data/<name>/`. Jeder enthält die Logging-CSVs
der Wärmemengenzähler nach dem Muster `logging_heat-energy_*.csv` und eine
Wetterdatei `open-meteo-*.csv`. Stage 1 findet beide über diese Namensmuster.

## `gebaeude_a/` — die anonymisierte Fassung der Originaldaten

Das ist die **Eingabe, aus der alle Zahlen und Abbildungen der Arbeit
entstehen**, und der Default-Datensatz aller Stages.

Die Messreihen stammen aus dem realen Betrieb eines Gewerbegebäudes. Ersetzt
sind ausschließlich die identifizierenden Angaben:

| | Original | hier |
|---|---|---|
| Gebäudekennung im Dateinamen | interne Kennung | `gebaeude-a` |
| Zählernummern im Spaltenkopf | Seriennummern der Geräte | `10000001` – `10000003` |
| Koordinaten der Wetterdatei | Standort des Gebäudes | auf die Stadtmitte gerundet |

**Messwerte, Zeitstempel und Spaltenreihenfolge sind unverändert** — andernfalls
wäre keine der berichteten Zahlen mehr nachrechenbar. Weil die Reihenfolge im
Dateikopf erhalten bleibt, vergibt `stage1_load.py` unverändert `wmz_1`,
`wmz_2` und `wmz_3`. Der Schritt ist als ausführbares Skript in
[`src/tools/anonymize_dataset.py`](../src/tools/anonymize_dataset.py)
festgehalten.

> **Nutzungsbedingungen.** Die Daten unterliegen dem Betriebs- und
> Geschäftsinteresse des Wärmeversorgers und stehen **nicht** unter der
> MIT-Lizenz des Codes. Sie sind allein zur Nachvollziehbarkeit dieser Arbeit
> beigelegt. Weiterverbreitung, kommerzielle Nutzung und jeder Versuch, Gebäude
> oder Betreiber zu reidentifizieren, sind nicht gestattet.

## `demo_synthetic/` — synthetischer Ersatz zum Ausprobieren

Nicht im Repository, in 15 Sekunden erzeugt:

```bash
python src/tools/make_synthetic_data.py     # schreibt data/demo_synthetic/
```

Nachgebildet ist allein das **Dateiformat**, alle Messwerte sind **erfunden**;
die Zahlen der Arbeit reproduziert dieser Datensatz nicht. Was genau
nachgebildet ist und was nicht, steht im Docstring von
[`src/tools/make_synthetic_data.py`](../src/tools/make_synthetic_data.py).

## Eigene Daten einhängen

Ordner `data/<name>/` anlegen, die Zähler-Exporte als
`logging_heat-energy_*.csv` und die Wetterdatei als `open-meteo-*.csv`
ablegen, Pipeline mit `--dataset <name>` starten. Die Zählernummern in den
Spaltenköpfen sind beliebig — ihre **Reihenfolge** legt `wmz_1`, `wmz_2`, …
fest. Details im [README](../README.md#eigenen-datensatz-hinzufügen).

Alles außer `gebaeude_a/` ist gitignored.
