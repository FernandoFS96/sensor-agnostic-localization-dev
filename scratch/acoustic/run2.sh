#!/usr/bin/env bash
# Step 1: the B1-winning regressor on B2 (no context, so the environment is unknown to it).
# Step 2: ablations toward the NPs' front end on B1:
#   pos -- same transformer, but the NPs' Fourier code of absolute positions + translation aug
#   np  -- the NPs' SpatialEncoder itself (paper settings) + MLP head, absolute frame
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
SCR=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=scripts/eval/acoustic_regressor.py
$PY $R train --cache $SCR/acoustic_cache_b2 --data-dir data/pathD/hermite_distinct/processed/geometry_split \
    --feat tone --steps 80000 --out-dir output/eval_paper/baselines/b2 > scratch/acoustic/train_b2_tone.log 2>&1 &
$PY $R train --cache $SCR/acoustic_cache --feat tone --pos-code fourier --frame absolute --aug trans \
    --steps 80000 --tag npos > scratch/acoustic/train_npos.log 2>&1 &
$PY $R train --cache $SCR/acoustic_cache --feat raw --encoder np --frame absolute --aug trans \
    --steps 80000 --tag npenc > scratch/acoustic/train_npenc.log 2>&1 &
wait
echo "=== ALL 3 DONE $(date '+%F %T') ==="
