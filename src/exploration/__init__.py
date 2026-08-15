"""Datenexplorations-Modul.

Sammelt eigenstaendige Explorations-Skripte, die unabhaengig von der
Detektions-Pipeline laufen und die Eigenschaften des Eingangssignals
visualisieren. Sie produzieren Figuren in ``outputs/<dataset>/figures/``,
die in der Masterarbeit als Beleg fuer Modellierungs- und Feature-
Entscheidungen verwendet werden koennen.

Abgrenzung zu ``src/tools/`` (Stage-spezifische Explorationen wie
``stage2_explore.py``, ``stage3_explore.py``):

* ``src/tools/`` enthaelt Skripte, die direkt auf einer Pipeline-Stage
  aufsetzen (z. B. ``stage2_explore.py`` visualisiert das Ergebnis von
  Stage 2) - sie heissen nach ihrer Stage und liefern stagebezogene
  Sanity-Checks.
* ``src/exploration/`` enthaelt **methoden-zentrierte** Auswertungen
  (Scatter mit Glaettung, Heatmaps, Autokorrelation, ...), die nicht
  einer einzelnen Stage zugeordnet sind, sondern allgemein die
  Daten-Eigenschaften herausarbeiten und so eigene Diskussions-
  abschnitte in der Thesis stuetzen.

Aktuelle Skripte:

* ``temperature_scatter.py`` - Scatter Verbrauch vs. Aussentemperatur
  mit LOWESS-Glaettung (Cleveland 1979).
* ``hour_weekday_heatmap.py`` - kombiniertes Stunden-x-Wochentag-
  Profil als Heatmap (Tukey 1977, EDA).
* ``autocorrelation.py`` - ACF/PACF auf den drei kW-Reihen
  (Box und Jenkins 1976), zur empirischen Bestaetigung der in Stage 3
  angesetzten Saisonalitaeten (24 h, 168 h).
* ``load_duration_curve.py`` - Jahresdauerlinien (absteigend sortierte
  Last) je WMZ; Grundlast vs. Spitzenlast (Verbruggen 1980).
* ``monthly_boxplot.py`` - monatliche kW-Verteilungen je WMZ als
  Boxplot-Reihe; Drift- und Strukturbruch-Diagnose (Tukey 1977).
* ``temperature_crosscorrelation.py`` - Kreuzkorrelation Verbrauch vs.
  Aussentemperatur ueber Lags 0..24 h; thermische Traegheit
  (Box und Jenkins 1976).
* ``stl_strength.py`` - numerische Trend-/Saison-Staerke aus der
  MSTL-Zerlegung je WMZ (Wang, Smith und Hyndman 2006).
* ``mutual_information.py`` - drei Masse (Pearson, Spearman, Mutual
  Information als Korrelationsaequivalent) fuer Verbrauch x
  Aussentemperatur auf einer gemeinsamen Skala; zeigt, wie stark der
  lineare Koeffizient die Abhaengigkeit unterschaetzt
  (Kraskov, Stoegbauer und Grassberger 2004).
* ``temperature_regimes.py`` - Niveau UND Streuung der Heizleistung je
  Temperaturklasse; beziffert den Knick an der Heizgrenze und die
  Regimeabhaengigkeit der Varianz (Hammarsten 1987).
"""
