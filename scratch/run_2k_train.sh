#!/usr/bin/env bash
# Step 3 only: the 2000-layout data is already generated + preprocessed.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
DD=data/random_task_2k/hermite_distinct/processed_delay/geometry_split
C="data=calib_bb data.data_dir=$DD model=spatial_cnp model.spatial.norm_acoustic=none \
   training.batch_size=128 training.num_workers=4 seed=2 wandb.enabled=false"
run(){ local n=$1; shift; echo "[t] $(date +%H:%M:%S) $n"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C "$@" \
    exp_name="$n" > "scratch/${n}.log" 2>&1 & }
run PFN_2k        model.spatial.use_position=false
sleep 25
run PFN_2k_aug    model.spatial.use_position=false data.trans_aug_m=500
sleep 25
run ORACLE_2k     model.spatial.use_position=true model.spatial.max_wavelength=4000
sleep 25
run PFN_2k_g100   model.spatial.use_position=false data.max_train_geometries=100
sleep 25
run PFN_2k_g400   model.spatial.use_position=false data.max_train_geometries=400
wait
echo "[t] $(date +%H:%M:%S) ALL COMPLETE"
echo "============ RESULTS (chance ~481 m, analytic 18.4 m) ============"
for n in PFN_2k_g100 PFN_2k_g400 PFN_2k PFN_2k_aug ORACLE_2k; do
  d=$(ls -d output/hydra/training/*/$n/seed_2 2>/dev/null | tail -1)
  if [ -n "$d" ] && [ -f "$d/train_log.csv" ]; then
    awk -F, -v N="$n" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}; lt=$5; le=$1}
      END{if(m=="")printf "  %-14s NO DATA (crashed)\n",N;
          else printf "  %-14s best val=%8.2f @ep%-5s | last train %.1f @ep%s\n",N,m,e,lt,le}' "$d/train_log.csv"
  else echo "  $n  MISSING"; fi
done
