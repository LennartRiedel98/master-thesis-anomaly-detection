#!/usr/bin/env bash
#
# Parallel-Lauf: LSTM-AE HPO Option A (gridwh) auf CPU - alle neun
# (Variante x WMZ)-Jobs GLEICHZEITIG, jeder in einem isolierten
# Temp-Dataset outputs/_par_<v>_<w>/ (Parquet/Scaler nur per Symlink).
# So nutzen wir die ~12 Kerne aus (ein rekurrentes LSTM allein laeuft
# faktisch single-threaded) -> ~6-8x schneller als der sequentielle Lauf.
#
# Jeder Job ist single-threaded gepinnt (OMP_NUM_THREADS=1), damit 9 Jobs
# sauber auf 9 Kerne gehen statt sich gegenseitig zu ueberzeichnen.
#
# Aufruf:
#   bash src/tools/run_parallel_lstm.sh            # voller Lauf
#   bash src/tools/run_parallel_lstm.sh 400        # Test: --max-train-rows 400
#
set -eo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
DS=demo_synthetic
REAL=outputs/$DS
TS=$(date +%Y%m%d_%H%M%S)
LOG=$REAL/hpo/parallel_${TS}.log
MAXROWS="${1:-}"
ROWARG=()
[ -n "$MAXROWS" ] && ROWARG=(--max-train-rows "$MAXROWS")

VARIANTS=(raw residual trend)
WMZ=(wmz_1 wmz_2 wmz_3)

mkdir -p "$REAL/hpo"
echo "=== Parallel-Start $(date) | rows=${MAXROWS:-voll} | Log $LOG ===" | tee "$LOG"

# Mac waehrend des Laufs wachhalten (caffeinate endet automatisch mit $$).
caffeinate -i -w $$ &

# Baseline der bisherigen (Random-Search-)Ergebnisse sichern.
cp "$REAL/hpo/best_hparams.json"             "$REAL/hpo/best_hparams_before_gridwh_par.json"    2>/dev/null || true
cp "$REAL/reports/stage10_metrics.csv"       "$REAL/reports/stage10_metrics_before_gridwh_par.csv" 2>/dev/null || true

pids=()
for v in "${VARIANTS[@]}"; do
  for w in "${WMZ[@]}"; do
    T=outputs/_par_${v}_${w}
    rm -rf "$T"; mkdir -p "$T"
    ln -sfn "../$DS/parquet" "$T/parquet"
    ln -sfn "../$DS/scalers" "$T/scalers"
    (
      export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
      "$PY" -u src/stage8_hpo.py      --dataset "_par_${v}_${w}" --models lstm_ae \
            --lstm-strategy gridwh --device cpu --variants "$v" --wmz "$w" "${ROWARG[@]}" \
      && "$PY" -u src/stage10_evaluate.py --dataset "_par_${v}_${w}" --models lstm_ae \
            --device cpu --variants "$v" --wmz "$w" "${ROWARG[@]}"
    ) > "outputs/_par_${v}_${w}/job.log" 2>&1 &
    pids+=($!)
    echo "  gestartet: $v/$w (PID $!)" | tee -a "$LOG"
  done
done

echo "  warte auf ${#pids[@]} parallele Jobs ..." | tee -a "$LOG"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== alle Jobs beendet ($fail Fehler) $(date) ===" | tee -a "$LOG"
[ "$fail" -ne 0 ] && echo "WARNUNG: $fail Job(s) fehlgeschlagen - siehe outputs/_par_*/job.log" | tee -a "$LOG"

# Single-threaded Merge ins echte Dataset (kein Race).
"$PY" src/tools/merge_parallel.py --dataset "$DS" 2>&1 | tee -a "$LOG"

# Temp-Datasets aufraeumen.
for v in "${VARIANTS[@]}"; do for w in "${WMZ[@]}"; do rm -rf "outputs/_par_${v}_${w}"; done; done

echo "=== Fertig $(date) ($fail Job-Fehler) ===" | tee -a "$LOG"
echo "Ergebnisse: $REAL/hpo/best_hparams.json + $REAL/reports/stage10_metrics.csv" | tee -a "$LOG"
