#!/usr/bin/env bash
# B2 checks on the plain regressor, beside the running dev-set PFN (AB_dist_pfnset):
#  1. robustness of the trained B2 regressor (Euclidean loss, no dropout): sensor-position
#     error 2/5 m and 3/5/7 of 10 sensors lost, all 50 points; plus the same position
#     errors for both B2 PFN runs (eval_calibration) so the two are scored alike;
#  2. the regressor with a Gaussian output (NLL, variance floor 0.01 m^2), 80k steps as
#     the original, then the same with train-time sensor dropout (p 0.3, min_keep 3) to
#     match the merged PFN recipe without context.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
SCR=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad
DD=data/pathD/hermite_distinct/processed/geometry_split
OUT=output/eval_paper/baselines/b2
R=scripts/eval/acoustic_regressor.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
echo "start $(date '+%F %T')"
$PY $R eval --cache $SCR/acoustic_cache_b2 --data-dir $DD --feat tone \
    --ckpt $OUT/acoustic_tone.pt --out-dir $OUT > scratch/acoustic/eval_reg_b2_robust.log 2>&1
O=$(ls -d output/hydra/training/*/PD_B_pfnset/seed_0 | tail -1)
N=$(ls -d output/hydra/training/*/PD_B_pfnset_fixdrop/seed_0 | tail -1)
for e in 2 5; do
  $PY scripts/eval/eval_calibration.py --data-dir $DD --models old=$O/last.pt fixdrop=$N/best.pt \
      --pos-err $e --out-dir output/eval_paper/b2_calibctx_set_pos$e > scratch/acoustic/eval_b2_pos$e.log 2>&1
done
echo "check 1 done $(date '+%F %T')"
$PY $R train --cache $SCR/acoustic_cache_b2 --data-dir $DD --feat tone --steps 80000 \
    --loss nll --tag nll --out-dir $OUT > scratch/acoustic/train_b2_tone_nll.log 2>&1
echo "nll done $(date '+%F %T')"
$PY $R train --cache $SCR/acoustic_cache_b2 --data-dir $DD --feat tone --steps 80000 \
    --loss nll --sensor-drop 0.3 --tag nll_drop --out-dir $OUT > scratch/acoustic/train_b2_tone_nll_drop.log 2>&1
echo "=== DONE $(date '+%F %T') ==="
