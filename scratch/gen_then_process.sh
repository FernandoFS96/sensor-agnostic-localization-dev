#!/usr/bin/env bash
# Wait for the broadband generation to finish, verify it succeeded, then run the
# geometry-split preprocessing. Chained so both stay in the background.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
GEN_PID="${1:-}"
ROOT=data/random_task_bb/hermite_distinct

echo "[chain] $(date +%H:%M:%S) waiting for generation (pid $GEN_PID)"
if [ -n "$GEN_PID" ]; then
  while kill -0 "$GEN_PID" 2>/dev/null; do sleep 30; done
fi
echo "[chain] $(date +%H:%M:%S) generation process ended"

if ! grep -q "^Done\." scratch/gen_broadband_full.log; then
  echo "[chain] ABORT: generation did not report Done. -- last lines:"
  tail -20 scratch/gen_broadband_full.log
  exit 1
fi
NSETS=$(ls -d $ROOT/position_set_* 2>/dev/null | wc -l)
echo "[chain] generation OK: $NSETS position sets, $(du -sh $ROOT | cut -f1)"

echo "[chain] $(date +%H:%M:%S) preprocessing (geometry split 70/15/15)"
$PY data/process_data.py --data-root "$ROOT" --mode geometry \
    --train-geoms 70 --val-geoms 15 --test-geoms 15
rc=$?
echo "[chain] $(date +%H:%M:%S) preprocessing exit=$rc"
if [ $rc -eq 0 ]; then
  SD=$ROOT/processed/geometry_split
  echo "[chain] outputs:"; ls -la "$SD" 2>/dev/null
  $PY - <<'PYEOF'
import pickle, numpy as np
m = pickle.load(open("data/random_task_bb/hermite_distinct/processed/geometry_split/metadata.pkl","rb"))
print(f"[chain] tau={m['tau']} n_sensors={m['n_sensors']} feat_dim={m['feat_dim']} thetas={m['thetas']}")
s = m['split']; print(f"[chain] split: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])} layouts")
PYEOF
fi
echo "[chain] $(date +%H:%M:%S) DONE"
