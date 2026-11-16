#!/usr/bin/env bash
# Context-conditioned regressor on B2 with the B2 PFN recipe (PD_B_pfnset_fixdrop),
# model swapped (MODEL=<config> selects another model config, TRAIN_DD=<dir> another dataset). Usage: run_ctxreg_b2.sh <seed> <exp_name> [extra overrides]
cd /home/fer/tesis/sensor-agnostic-underwater-localization
seed=$1; name=$2; shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date '+%F %T')"
TQDM_DISABLE=1 /home/fer/miniconda3/envs/torch/bin/python scripts/train/train_np_geometry.py \
    data.data_dir=${TRAIN_DD:-data/pathD/hermite_distinct/processed/geometry_split} \
    data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
    model=${MODEL:-spatial_ctxreg} data.rot_aug=true \
    data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
    training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=$name seed=$seed "$@"
echo "=== DONE $(date '+%F %T') ==="
