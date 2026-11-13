#!/usr/bin/env bash
# PFN encoder size sweep on B2 (PD_B_pfnset_fixdrop recipe: set encoder, centroid frame,
# rot aug, var floor 0.01 m^2, bernoulli sensor dropout, calibration context, 200 epochs).
# Known: width 256 / 6 layers = 4.14 m (6.11 M), width 128 / 6 layers = 6.72 m (2.47 M).
#   d256_l3  depth vs width at the same width
#   d192_l6  between the two known widths
#   s1       the current configuration, seed 1 (seed noise)
#   d384_l6  is the current size already enough?
# The two small ones run together; the others alone (GPU memory). Then every model is
# scored with the calibration context and 0/3/5/7 of 10 sensors lost.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
DD=data/pathD/hermite_distinct/processed/geometry_split
train() {   # exp_name, seed, extra overrides...
  name=$1; seed=$2; shift 2
  $PY scripts/train/train_np_geometry.py data.data_dir=$DD \
      data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
      model=spatial_pfn_set model.readout=joint data.rot_aug=true \
      data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
      training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=$name seed=$seed "$@" \
      > scratch/acoustic/${name}_s$seed.log 2>&1
  echo "$name s$seed done $(date '+%F %T') (exit $?)"
}
echo "start $(date '+%F %T')"
train PD_B_pfnset_d256_l3 0 model.spatial.set_layers=3 &
train PD_B_pfnset_d192_l6 0 model.spatial.set_d_model=192 &
wait
train PD_B_pfnset_fixdrop 1
train PD_B_pfnset_d384_l6 0 model.spatial.set_d_model=384
ck() { ls -d output/hydra/training/*/$1/seed_$2 | tail -1; }
M="d256_l6_s0=$(ck PD_B_pfnset_fixdrop 0)/best.pt d256_l6_s1=$(ck PD_B_pfnset_fixdrop 1)/best.pt"
M="$M d128_l6=$(ck PD_B_pfnset_d128 0)/best.pt d256_l3=$(ck PD_B_pfnset_d256_l3 0)/best.pt"
M="$M d192_l6=$(ck PD_B_pfnset_d192_l6 0)/best.pt d384_l6=$(ck PD_B_pfnset_d384_l6 0)/best.pt"
for k in 0 3 5 7; do
  $PY scripts/eval/eval_calibration.py --data-dir $DD --models $M --drop $k \
      --out-dir output/eval_paper/b2_pfn_size_sweep_drop$k > scratch/acoustic/eval_b2_sweep_drop$k.log 2>&1
done
echo "=== DONE $(date '+%F %T') ==="
