#!/usr/bin/env bash
# Lightweight sensor-set front end. Width 128 is known to cost +0.22 m at 1.25 M
# parameters (abl_d128). Here: (1) three smaller B1 regressors built the new way
# (feed-forward kept, MLP tokens, readout token), (2) the width-128 encoder inside
# the PFN on B2 with the PD_B_pfnset_fixdrop recipe, scored like it (calibration
# context, 0/3/5/7 sensors lost) beside fixdrop.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
export PY=/home/fer/miniconda3/envs/torch/bin/python
export CACHE=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
DD=data/pathD/hermite_distinct/processed/geometry_split
echo "start $(date '+%F %T')"
(
  $PY scripts/train/train_np_geometry.py data.data_dir=$DD \
      data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
      model=spatial_pfn_set model.readout=joint model.spatial.set_d_model=128 data.rot_aug=true \
      data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
      training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=PD_B_pfnset_d128 seed=0 \
      > scratch/acoustic/PD_B_pfnset_d128_s0.log 2>&1
  echo "pfn d128 train done $(date '+%F %T') (exit $?)"
  F=$(ls -d output/hydra/training/*/PD_B_pfnset_fixdrop/seed_0 | tail -1)
  N=$(ls -d output/hydra/training/*/PD_B_pfnset_d128/seed_0 | tail -1)
  for k in 0 3 5 7; do
    $PY scripts/eval/eval_calibration.py --data-dir $DD --models fixdrop=$F/best.pt d128=$N/best.pt \
        --drop $k --out-dir output/eval_paper/b2_calibctx_set_d128_drop$k > scratch/acoustic/eval_b2_d128_drop$k.log 2>&1
  done
  echo "pfn d128 eval done $(date '+%F %T')"
) &
run() {   # tag, extra flags...
  tag=$1; shift
  $PY scripts/eval/acoustic_regressor.py train --cache $CACHE --feat raw --steps 80000 "$@" \
      > scratch/acoustic/train_raw_$tag.log 2>&1
  echo "$tag done $(date '+%F %T') (exit $?)"
}
export -f run
printf '%s\n' "light_d128_l3 --tag light_d128_l3 --d 128 --layers 3" \
              "light_d96_l4 --tag light_d96_l4 --d 96 --layers 4" \
              "light_d64_l6 --tag light_d64_l6 --d 64 --layers 6" \
  | xargs -P 3 -I{} bash -c 'run {}'
wait
echo "=== DONE $(date '+%F %T') ==="
