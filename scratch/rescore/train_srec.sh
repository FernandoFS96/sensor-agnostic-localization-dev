#!/usr/bin/env bash
# S-RecCNP on the development set (2026-11-11), for the AOR paper's same-data
# recurrent pair: RecCNP vs S-RecCNP (Table 3, shared variant) and the S-RecCNP
# arm of the trajectory A/B (Table 4, distinct variant).
#
# Recipe = the development-set family runs (paper_spatial_ranp / AB_dist_*):
# layout_ood, num_hidden 128, per_sample false, trans_aug 150 m for spatial
# models. One run at a time: each holds ~55 GB of RAM (49 GB train pickle), and
# the test-only re-score runs alongside.
#
# Known difference from the paper's runs: best.pt now follows every validation
# improvement; the paper's runs only saved on improvements > 0.3 m
# (early_stop_min_delta). Patience/stopping is unchanged.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
LOG=scratch/rescore; mkdir -p $LOG

COMMON="data=layout_ood training.num_workers=4 wandb.offline=true wandb.enabled=false
        model.num_hidden=128 data.context.per_sample=false model=spatial_rcnp data.trans_aug_m=150"

run () {  # $1=exp $2=seed $3=extra overrides
  echo "  -> $1 s$2  $(date '+%F %T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $COMMON $3 seed=$2 exp_name=$1 \
      > $LOG/$1_s$2.log 2>&1
  echo "     rc=$?  $(grep -h 'done. best val mae' $LOG/$1_s$2.log | sed 's/ -> .*//')  $(date '+%F %T')"
}

echo "=== S-RecCNP, shared variant  $(date '+%F %T') ==="
for S in 0 1 2; do run paper_spatial_rcnp $S "dataset=random_hermite_shared"; done
touch $LOG/SHARED_DONE

echo "=== S-RecCNP, distinct variant  $(date '+%F %T') ==="
for S in 0 1 2; do
  run AB_dist_spatial_rcnp $S "dataset=random_hermite_distinct data.data_dir=data/random_task/hermite_distinct/processed_common/geometry_split"
done
touch $LOG/DISTINCT_DONE
echo "=== TRAINING COMPLETE  $(date '+%F %T') ==="
