#!/usr/bin/env bash
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
DD=data/random_task_2k/hermite_distinct/processed_delay/geometry_split
C="data=calib_bb data.data_dir=$DD model=spatial_cnp model.spatial.norm_acoustic=none \
   model.spatial.use_position=false training.batch_size=128 training.num_workers=4 \
   seed=2 wandb.enabled=false"
run(){ local n=$1; shift; echo "[sid] $(date +%H:%M:%S) $n"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C "$@" exp_name="$n" \
    > "scratch/${n}.log" 2>&1 & }
run PFN_2k_sid   model.spatial.sensor_id_embed=true
sleep 25
run PFN_2k_flat  model.spatial.tokenize=false
wait
echo "[sid] $(date +%H:%M:%S) COMPLETE"
echo "===== RESULTS (chance 481; PFN_2k baseline 427; ORACLE_2k 102; analytic 18.4) ====="
for n in PFN_2k_sid PFN_2k_flat; do
  d=$(ls -d output/hydra/training/*/$n/seed_2 2>/dev/null | tail -1)
  [ -f "$d/train_log.csv" ] && awk -F, -v N="$n" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}; lt=$5}
    END{printf "  %-14s best val=%8.2f @ep%-5s (train %.1f)\n",N,m,e,lt}' "$d/train_log.csv"
done
