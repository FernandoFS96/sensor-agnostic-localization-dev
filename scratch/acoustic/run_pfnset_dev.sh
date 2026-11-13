#!/usr/bin/env bash
# Development set, distinct arm of the trajectory A/B: the new PFN (sensor-set
# encoder, centroid frame, rotation augmentation, variance floor 0.01 m^2, sensor
# dropout) trained exactly like AB_dist_spatial_cnp (prefix regime, 140 training
# deployments x 350 transits, same split as the shared arm), at the recipe's 200
# epochs and lr 1e-4. Then:
#   - eval_suite tiers 1-4 (the paper's dev-set protocol: region MAE, noise and
#     order controls) -> output/eval_paper/ab_distinct_pfnset, to set beside
#     ab_distinct (S-AttnCNP 20.31 m, noise 22.8);
#   - eval_calibration (context from another transit), new PFN and the paper's
#     S-AttnCNP seeds -> output/eval_paper/dev_distinct_calibctx.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/random_task/hermite_distinct/processed_common/geometry_split
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date '+%F %T')"
TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py \
    data=layout_ood dataset=random_hermite_distinct training.num_workers=4 \
    wandb.offline=true wandb.enabled=false data.data_dir=$DD \
    model=spatial_pfn_set model.readout=joint data.rot_aug=true \
    data.context.per_sample=true data.sensor_drop.mode=bernoulli \
    training.epochs=200 training.lr=1e-4 exp_name=AB_dist_pfnset seed=0 \
    > scratch/acoustic/AB_dist_pfnset_s0.log 2>&1
echo "train done $(date '+%F %T')"
N=$(ls -d output/hydra/training/*/AB_dist_pfnset/seed_0 | tail -1)
TQDM_DISABLE=1 $PY -u scripts/eval/eval_suite.py --models pfnset=$N/best.pt \
    --tiers 1,2,3,4 --data-dir $DD --out-dir output/eval_paper/ab_distinct_pfnset \
    > output/eval_paper/ab_distinct_pfnset.log 2>&1
echo "eval_suite done $(date '+%F %T')"
O=output/hydra/training
TQDM_DISABLE=1 $PY scripts/eval/eval_calibration.py --data-dir $DD \
    --models pfnset=$N/best.pt \
             spatial_cnp=$O/2026-10-03/AB_dist_spatial_cnp/seed_0/best.pt \
             spatial_cnp=$O/2026-10-01/AB_dist_spatial_cnp/seed_1/best.pt \
             spatial_cnp=$O/2026-10-01/AB_dist_spatial_cnp/seed_2/best.pt \
    --out-dir output/eval_paper/dev_distinct_calibctx > scratch/acoustic/eval_dev_calib.log 2>&1
echo "=== DONE $(date '+%F %T') ==="
