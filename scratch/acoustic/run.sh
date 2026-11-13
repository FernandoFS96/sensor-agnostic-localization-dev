#!/usr/bin/env bash
# Option-3 diagnostic: can a plain network use B1's acoustics as well as the MFP?
# Two front ends of the same sensor-set regressor, run side by side on the GPU.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
CACHE=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in tone raw; do
  $PY scripts/eval/acoustic_regressor.py train --cache $CACHE --feat $f --steps 80000 \
      > scratch/acoustic/train_$f.log 2>&1 &
  sleep 60
done
wait
echo "=== BOTH DONE $(date '+%F %T') ==="
