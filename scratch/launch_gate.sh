#!/usr/bin/env bash
# Detached gate experiment on the broadband dataset.
#   PFN    = position-free + CALIBRATION context  <- the experiment
#   ORACLE = position-given + CALIBRATION context <- the ceiling on this physics
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
DD=data/random_task_bb/hermite_distinct/processed/geometry_split
COMMON="data.context.calibration=true data.trans_aug_m=0 data.data_dir=$DD \
        training.batch_size=128 training.num_workers=4 seed=2"

echo "[gate] $(date) launching PFN (position-free)"
TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py \
  model=spatial_cnp model.spatial.use_position=false $COMMON \
  exp_name=PFN_calib_nopos_bb_gate > scratch/gate_pfn.log 2>&1 &
P1=$!
sleep 120   # stagger so the two 16.8 GB pickle loads do not collide
echo "[gate] $(date) launching ORACLE (position-given)"
TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py \
  model=spatial_cnp model.spatial.use_position=true $COMMON \
  exp_name=ORACLE_calib_pos_bb > scratch/gate_oracle.log 2>&1 &
P2=$!
echo "[gate] pids: PFN=$P1 ORACLE=$P2"
wait $P1; echo "[gate] $(date) PFN exit=$?"
wait $P2; echo "[gate] $(date) ORACLE exit=$?"
echo "[gate] $(date) BOTH COMPLETE"
for d in output/hydra/training/*/PFN_calib_nopos_bb_gate/seed_2 \
         output/hydra/training/*/ORACLE_calib_pos_bb/seed_2; do
  [ -f "$d/train_log.csv" ] && awk -F, -v D="$d" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}}
    END{print "[gate] "D": best val_mae="m" @ epoch "e}' "$d/train_log.csv"
done
