#!/usr/bin/env bash
# Follow-up to run_frontend_ablation.sh: one choice at a time costs 0-0.8 m (sum ~1.4 m,
# ~2.3 m with the position handling of npos), but the NPs' SpatialEncoder is 20 m worse.
# So either the choices interact, or the gap is in what the leave-one-out did not copy
# (the legacy block: post-norm, concat-residual linear, dropout 0.1, 4 heads).
#   stack5      all five set-encoder choices at once (d 128, 3 layers, linear token,
#               no FFN, attention pool), centroid frame, rotation aug
#   stack5_pos  the same + the NPs' position handling (Fourier code of absolute
#               positions, translation aug): every item of the list put back
#   npenc_cent  the SpatialEncoder itself with the new position handling (centroid
#               frame, rotation aug): does the legacy block work once positions are fixed?
cd /home/fer/tesis/sensor-agnostic-underwater-localization
export PY=/home/fer/miniconda3/envs/torch/bin/python
export CACHE=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date '+%F %T')"
run() {   # tag, extra flags...
  tag=$1; shift
  $PY scripts/eval/acoustic_regressor.py train --cache $CACHE --feat raw --steps 80000 "$@" \
      > scratch/acoustic/train_raw_$tag.log 2>&1
  echo "$tag done $(date '+%F %T') (exit $?)"
}
export -f run
S5="--d 128 --layers 3 --tokenizer linear --no-ffn --summary pool"
printf '%s\n' "stack5 --tag stack5 $S5" \
              "stack5_pos --tag stack5_pos $S5 --pos-code fourier --frame absolute --aug trans" \
              "npenc_cent --tag npenc_cent --encoder np --frame centroid --aug rot" \
  | xargs -P 3 -I{} bash -c 'run {}'
echo "=== DONE $(date '+%F %T') ==="
