# data/

Dieser Ordner ist im Repository **leer**. Er nimmt die Eingabedaten auf, je
Datensatz ein Unterordner `data/<name>/`.

## Es liegen hier keine echten Messdaten

Die Arbeit wurde auf realen Betriebsdaten eines Gewerbegebäudes gerechnet. Die
unterliegen dem Betriebs- und Geschäftsinteresse des Wärmeversorgers und sind
**nicht Teil der Veröffentlichung** — Begründung und Bezugsweg im
[README, Abschnitt „Daten"](../README.md#daten).

## Demo-Datensatz erzeugen

```bash
python src/tools/make_synthetic_data.py     # schreibt data/demo_synthetic/
```

Das legt fünf Logging-CSVs und eine Open-Meteo-CSV im **identischen Format**
der Originaldateien an — mit **erfundenen Messwerten**. Der Lauf dauert rund
15 Sekunden, erzeugt etwa 140 MB und ist bei gleichem Zufallsstartwert
byte-identisch wiederholbar. Deshalb steht im Repository der Generator und
nicht sein Ergebnis; `data/*/` ist bewusst gitignored.

Was nachgebildet ist und was nicht, steht im Docstring von
[`src/tools/make_synthetic_data.py`](../src/tools/make_synthetic_data.py).

## Eigene Daten einhängen

Ordner `data/<name>/` anlegen, die Zähler-Exporte als
`logging_heat-energy_*.csv` und die Wetterdatei als `open-meteo-*.csv`
ablegen, Pipeline mit `--dataset <name>` starten. Als Formatvorlage dient der
erzeugte Demo-Datensatz. Details im
[README](../README.md#eigenen-datensatz-hinzufügen).
