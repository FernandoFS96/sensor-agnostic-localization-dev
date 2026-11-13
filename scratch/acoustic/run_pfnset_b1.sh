#!/usr/bin/env bash
# Step 3 check: the paper's calibration-regime B1 PFN (CL_B_pfn) with the new
# sensor-set front end, array-centered labels and rotation augmentation.
# Shorter schedule than the paper runs (200 epochs at lr 1e-4 instead of 1000 at
# 5e-5): the stand-alone regressor converged in the equivalent of ~50 epochs.
# Starts when the step 1-2 diagnostics (run2.sh) have finished.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
until grep -q "ALL 3 DONE" scratch/acoustic/run2_out.log 2>/dev/null; do sleep 60; done
echo "start $(date '+%F %T')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TQDM_DISABLE=1 /home/fer/miniconda3/envs/torch/bin/python scripts/train/train_np_geometry.py \
    data.data_dir=data/pfn10kd/hermite_distinct/processed/geometry_split \
    data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
    model=spatial_pfn_set model.readout=joint data.rot_aug=true \
    data.context.per_sample=true data.context.calibration=true \
    training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=CL_B_pfnset seed=0 \
    > scratch/acoustic/CL_B_pfnset_s0.log 2>&1
echo "=== DONE $(date '+%F %T') ==="
