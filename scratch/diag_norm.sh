#!/usr/bin/env bash
# The features are a SPARSE SPIKE TRAIN (1.27% of taps active) spanning 4 orders
# of magnitude, where the cue is the spike INDEX (delay), not its amplitude.
# norm_acoustic=standardize (a single GLOBAL scalar, chosen for the old tone data
# where amplitude was the only cue) leaves per-record magnitude spread over ~280x.
# layernorm normalises each sensor record individually -> amplitude-invariant,
# spike position stands out. Test whether that lets the ORACLE FIT at all.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
DD=data/random_task_bb/hermite_distinct/processed/geometry_split
C="data.context.calibration=true data.trans_aug_m=0 data.data_dir=$DD \
   training.batch_size=128 training.num_workers=4 training.epochs=120 \
   training.early_stop_patience=0 model.spatial.use_position=true seed=2 wandb.enabled=false"
TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py model=spatial_cnp \
  model.spatial.norm_acoustic=layernorm $C \
  hydra.run.dir=scratch/diag_ln > scratch/diag_ln.log 2>&1 &
A=$!
sleep 90
TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py model=spatial_cnp \
  model.spatial.norm_acoustic=layernorm training.lr=3e-4 $C \
  hydra.run.dir=scratch/diag_ln_lr > scratch/diag_ln_lr.log 2>&1 &
B=$!
wait $A; wait $B
echo "[diag] done"
for d in scratch/diag_ln scratch/diag_ln_lr; do
  awk -F, -v D="$d" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}}
    END{print D" best val="m" @ep"e}' $d/train_log.csv
  echo "   $(tail -1 $d/train_log.csv | cut -d, -f1,5,9)  (ep,train_mae,val_mae)"
done
