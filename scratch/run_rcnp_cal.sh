#!/usr/bin/env bash
# S-RecCNP in the CALIBRATION regime on B1 (clean layout set) and B2 (Path D), three
# seeds each: the missing like-for-like row of the B1/B2 tables. Same settings as the
# CL_A_rcnp / PD_A_rcnp prefix runs (run_seeds.sh) plus calibration=true. The recurrent
# model reads one sequence, the calibration points then the target transit, so one
# context size per batch (per_sample=false), unlike the split models' PD_B/CL_B runs.
#
# GPU MEMORY: the sequence is up to 40 + 50 steps, so a run holds up to ~5.5 GB of the
# 16 GB card, and the card is already saturated by two runs. The first launch put four
# runs next to the two depth arms and the B1 seed ran out of memory. So: one recurrent
# run at a time while the depth arms (run_depth_cal.sh) train, two once they finish.
# PD_B_rcnp seed 0 was started by the first launch and is still running (WAIT_PID).
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD_D=data/pathD/hermite_distinct/processed/geometry_split
DD_C=data/pfn10kd/hermite_distinct/processed/geometry_split
LOG=scratch/seeds; mkdir -p $LOG
WAIT_PID=${WAIT_PID:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="data.max_train_geometries=null data.max_train_trajectories=null
      data.max_val_trajectories=null wandb.enabled=false"
CAL="data.context.per_sample=false data.context.calibration=true"

limit () {   # 1 while the depth arms train, then 2
  grep -q "ALL 12 RUNS DONE" scratch/depth/run_cal_out.log 2>/dev/null && echo 2 || echo 1
}
running () {
  local n; n=$(jobs -rp | wc -l)
  [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null && n=$((n + 1))
  echo $n
}
run () {   # $1=exp $2=model $3=extra $4=seed $5=data_dir
  while [ "$(running)" -ge "$(limit)" ]; do sleep 60; done
  echo "  -> $1 s$4  $(date '+%F %T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE data.data_dir=$5 \
      model=$2 $3 seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
  sleep 300    # let it reach its working memory before the next check
}

echo "=== S-RecCNP CALIBRATION REGIME (relaunch) — START $(date '+%F %T') ==="
run PD_B_rcnp spatial_rcnp "$CAL" 1 $DD_D
run CL_B_rcnp spatial_rcnp "$CAL" 0 $DD_C
run PD_B_rcnp spatial_rcnp "$CAL" 2 $DD_D
run CL_B_rcnp spatial_rcnp "$CAL" 1 $DD_C
run CL_B_rcnp spatial_rcnp "$CAL" 2 $DD_C
wait
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== ALL 6 RUNS DONE $(date '+%F %T') ==="
for f in $LOG/CL_B_rcnp_s*.log $LOG/PD_B_rcnp_s*.log; do
  printf "%-20s %s\n" "$(basename $f .log)" "$(grep -o 'done. best val mae.*@ epoch [0-9]*' $f || echo 'NO FINISH LINE')"
done
