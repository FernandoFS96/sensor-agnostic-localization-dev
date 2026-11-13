#!/usr/bin/env bash
# B2 run with both recipe fixes and sensor dropout: variance floor 0.01 m^2
# (var_floor_phys in spatial_pfn_set.yaml), best.pt tracking every val improvement
# (trainer), and train-time sensor dropout (bernoulli p=0.3, min_keep 3, as in the
# RANP topologies work). Otherwise identical to PD_B_pfnset. Starts when
# run_pfnset_b2.sh has finished, then scores both runs with every sensor and with
# 3/5/7 of 10 sensors lost (calibration context, MAE + ellipse coverage).
cd /home/fer/tesis/sensor-agnostic-underwater-localization
until grep -q "=== DONE" scratch/acoustic/run_pfnset_b2_out.log 2>/dev/null; do sleep 60; done
echo "start $(date '+%F %T')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TQDM_DISABLE=1 /home/fer/miniconda3/envs/torch/bin/python scripts/train/train_np_geometry.py \
    data.data_dir=data/pathD/hermite_distinct/processed/geometry_split \
    data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
    model=spatial_pfn_set model.readout=joint data.rot_aug=true \
    data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
    training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=PD_B_pfnset_fixdrop seed=0 \
    > scratch/acoustic/PD_B_pfnset_fixdrop_s0.log 2>&1
echo "train done $(date '+%F %T')"
O=$(ls -d output/hydra/training/*/PD_B_pfnset/seed_0 | tail -1)
N=$(ls -d output/hydra/training/*/PD_B_pfnset_fixdrop/seed_0 | tail -1)
for k in 0 3 5 7; do
  TQDM_DISABLE=1 /home/fer/miniconda3/envs/torch/bin/python scripts/eval/eval_calibration.py \
      --data-dir data/pathD/hermite_distinct/processed/geometry_split \
      --models old=$O/last.pt fixdrop=$N/best.pt --drop $k \
      --out-dir output/eval_paper/b2_calibctx_set_drop$k > scratch/acoustic/eval_b2_drop$k.log 2>&1
done
echo "=== DONE $(date '+%F %T') ==="
