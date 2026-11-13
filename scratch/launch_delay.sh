#!/usr/bin/env bash
# Gate experiment on DELAY features + fixed context sizes.
# Analytic reference on these exact features: geometry 6.8 m -> source 18.4 m.
# Chance = 481 m.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
C="data=calib_bb model=spatial_cnp model.spatial.norm_acoustic=none \
   training.batch_size=128 training.num_workers=4 seed=2"
run () {  # name, extra args
  local n=$1; shift
  echo "[run] $(date +%H:%M:%S) $n"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C "$@" \
    exp_name="$n" > "scratch/${n}.log" 2>&1 &
}
run PFN_delay_calib      model.spatial.use_position=false
sleep 20
run ORACLE_delay_calib   model.spatial.use_position=true
sleep 20
run CONTROL_delay_sametraj model.spatial.use_position=true data.context.calibration=false
sleep 20
run PFN_delay_calib_lr3e4 model.spatial.use_position=false training.lr=3e-4
wait
echo "[run] $(date +%H:%M:%S) ALL COMPLETE"
echo "=============== RESULTS (chance 481 m; analytic target 18.4 m) ==============="
for n in PFN_delay_calib ORACLE_delay_calib CONTROL_delay_sametraj PFN_delay_calib_lr3e4; do
  d=$(ls -d output/hydra/training/*/$n/seed_2 2>/dev/null | tail -1)
  [ -n "$d" ] && awk -F, -v N="$n" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}}
     END{printf "  %-26s best val_mae=%8.2f @ ep %s\n", N, m, e}' "$d/train_log.csv"
done
