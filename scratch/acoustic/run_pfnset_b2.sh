#!/usr/bin/env bash
# B2 version of the step 3 check: the paper's calibration-regime B2 PFN (PD_B_pfn)
# with the sensor-set front end, array-centered labels and rotation augmentation.
# Same recipe as CL_B_pfnset (200 epochs at lr 1e-4). The no-context regressor
# scores 3.88 m (all 50 points) on B2; this run tests whether calibration context
# adds anything on top of it when the environment varies per deployment.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
echo "start $(date '+%F %T')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TQDM_DISABLE=1 /home/fer/miniconda3/envs/torch/bin/python scripts/train/train_np_geometry.py \
    data.data_dir=data/pathD/hermite_distinct/processed/geometry_split \
    data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
    model=spatial_pfn_set model.readout=joint data.rot_aug=true \
    data.context.per_sample=true data.context.calibration=true \
    training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=PD_B_pfnset seed=0 \
    > scratch/acoustic/PD_B_pfnset_s0.log 2>&1
echo "=== DONE $(date '+%F %T') ==="
