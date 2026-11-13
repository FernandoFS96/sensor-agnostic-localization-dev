#!/usr/bin/env bash
# B2 depth-stratified study, CALIBRATION regime. Same arms as run_depth.sh (same
# train_{shallow,deep}.json, same untouched val/test), but trained with context
# drawn from another trajectory of the same deployment and theta, as the PD_B_*
# runs are. The prefix-regime arms carry the trajectory shortcut (a no-acoustics
# net reaches 24 m on B2 under a prefix, flat across depth), so their depth
# ratios and model ranking cannot be read as properties of the architectures.
# The full arm is PD_B_* (reused). The S-RecCNP arms wait for calibration
# support in the recurrent models.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
LOG=scratch/depth; mkdir -p $LOG
MAXJOBS=2        # pathD's train pickle is 7.2 GB; two runs sit near 25 G of 125 G

BASE="data.data_dir=data/pathD/hermite_distinct/processed/geometry_split
      data.max_train_geometries=null data.max_train_trajectories=null
      data.max_val_trajectories=null wandb.enabled=false
      data.context.per_sample=true data.context.calibration=true"

run () {   # $1=exp  $2=model-overrides  $3=arm  $4=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE $2 \
      data.train_geometry_ids=scratch/depth/train_$3.json \
      seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

echo "=== DEPTH-STRATIFIED B2, CALIBRATION REGIME — START $(date '+%F %T') ==="
for S in 0 1 2; do
  for ARM in shallow deep; do
    run DS_cal_${ARM}_pfn  "model=spatial_pfn model.readout=joint"   $ARM $S
    run DS_cal_${ARM}_cnpF "model=spatial_cnp model.attn_ffn=true"   $ARM $S
  done
done
wait
echo "=== ALL 12 RUNS DONE $(date '+%F %T') ==="
for f in $LOG/DS_cal_*_s*.log; do
  printf "%-28s %s\n" "$(basename $f .log)" "$(grep -o 'done. best val mae.*' $f || echo 'NO FINISH LINE')"
done
