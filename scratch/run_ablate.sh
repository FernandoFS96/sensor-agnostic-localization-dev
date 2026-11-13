#!/usr/bin/env bash
# 2x2 ablation at 8000x10: separate POSITIONS from the NORMALISATION they need.
#
# The first attempt at a one-variable control (positions ON, norm_acoustic=none)
# collapsed to chance -- train MAE stuck at 404 against position-free's 160, val
# drifting 489 -> 520 -- which _spatial.yaml already warns about: "none
# (collapses)". Unnormalised positions on top of unnormalised acoustics simply
# does not train. Positions therefore CANNOT be toggled on their own:
# standardisation has to come with them, and that makes normalisation a second
# variable which needs its own control before any gap can be attributed to
# position information.
#
#   PC  (have)      pos OFF, norm=none,        tokenize=false -> 102.8 m
#   PFS (this)      pos OFF, norm=standardize, tokenize=false -> normalisation alone
#   PAS (this)      pos ON,  norm=standardize, tokenize=false -> + positions
#   PA  (running)   pos ON,  norm=standardize, tokenize=true  -> + tokenisation
#
# Reading down that ladder attributes the total gap to each ingredient in turn.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
LOG=scratch/posaware
mkdir -p $LOG
MAXJOBS=3      # the main grid is still running; leave it GPU headroom

B="data=calib_bb data.data_dir=data/calib_10k/cells/g8000_t10 model=spatial_cnp
   model.spatial.tokenize=false model.spatial.norm_acoustic=standardize
   data.max_val_trajectories=10 data.max_train_geometries=8000
   data.max_train_trajectories=10 training.batch_size=128
   training.num_workers=4 wandb.enabled=false"

run () {   # $1=exp_name $2=use_position $3=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
  echo "  -> $1 s$3 $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $B \
      model.spatial.use_position=$2 seed=$3 exp_name=$1 \
      > $LOG/$1_s$3.log 2>&1 &
}

echo "=== ABLATION START $(date '+%F %T') ==="
for S in 1 2 3; do run PAS_g8000_t10 true  $S; done
for S in 1 2 3; do run PFS_g8000_t10 false $S; done
wait
echo "=== ABLATION COMPLETE $(date '+%F %T') ==="
