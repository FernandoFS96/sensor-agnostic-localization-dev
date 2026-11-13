#!/usr/bin/env bash
# Which part of the new sensor-set front end carries the gain over the NPs' SpatialEncoder?
# B1, no-context regressor on raw records (acoustic_regressor.py), 80k steps, the recipe of
# the 3.01 m baseline (acoustic_raw: width 256, 6 pre-norm transformer layers with FFN,
# MLP tokens, readout token, centroid frame, rotation aug). Leave-one-out: each run puts
# back ONE SpatialEncoder choice. Plus the baseline with seed 1 as the seed-noise control.
# Two runs at a time; results in output/eval_paper/baselines/b1/results_acoustic_raw_abl_*.csv
cd /home/fer/tesis/sensor-agnostic-underwater-localization
export PY=/home/fer/miniconda3/envs/torch/bin/python
export CACHE=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache
echo "start $(date '+%F %T')"
run() {   # tag, extra flags...
  tag=$1; shift
  $PY scripts/eval/acoustic_regressor.py train --cache $CACHE --feat raw --steps 80000 "$@" \
      > scratch/acoustic/train_raw_$tag.log 2>&1
  echo "$tag done $(date '+%F %T') (exit $?)"
}
export -f run
printf '%s\n' "abl_noffn --tag abl_noffn --no-ffn" \
              "abl_pool --tag abl_pool --summary pool" \
              "abl_lintok --tag abl_lintok --tokenizer linear" \
              "s1 --seed 1" \
              "abl_d128 --tag abl_d128 --d 128" \
              "abl_l3 --tag abl_l3 --layers 3" \
  | xargs -P 2 -I{} bash -c 'run {}'
echo "=== DONE $(date '+%F %T') ==="
