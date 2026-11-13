#!/usr/bin/env bash
# SCREEN A -- does the position-free arm benefit from more layouts, now that the
# architecture actually works? The old curve (489/458/427 at 100/400/1600) was
# measured with the JOINT readout, which we later proved is structurally broken
# for this task (tokenize=false took it 427 -> 180). So that curve tells us
# nothing about whether data helps. Re-run it with the WORKING architecture.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
DD=data/random_task_2k/hermite_distinct/processed_delay/geometry_split
C="data=calib_bb data.data_dir=$DD model=spatial_cnp model.spatial.norm_acoustic=none \
   model.spatial.use_position=false model.spatial.tokenize=false \
   training.batch_size=128 training.num_workers=4 seed=2 wandb.enabled=false"
for G in 100 400 1600; do
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C \
      data.max_train_geometries=$G exp_name=SCRA_flat_g${G} \
      > scratch/SCRA_flat_g${G}.log 2>&1 &
done
wait
echo "SCREEN A COMPLETE"
echo "===== position-free, WORKING architecture (tokenize=false), chance 481 ====="
for G in 100 400 1600; do
  d=$(ls -d output/hydra/training/*/SCRA_flat_g${G}/seed_2 2>/dev/null | tail -1)
  [ -f "$d/train_log.csv" ] && awk -F, -v G="$G" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}; lt=$5}
    END{printf "  %5s layouts : best val=%7.2f @ep%-5s (train %.1f)\n",G,m,e,lt}' "$d/train_log.csv"
done
echo "  (old curve with the BROKEN joint readout: 489 / 458 / 427)"
