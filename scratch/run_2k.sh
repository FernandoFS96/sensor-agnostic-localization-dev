#!/usr/bin/env bash
# Prior-size experiment: 2000 layouts x 10 trajectories (same sim budget as
# 100 x 100, but 20x the LAYOUT diversity -- the axis the ablation says we are
# starved on: 10->626 m, 20->439, 40->384, 70->281, still falling).
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
RAW=./data/random_task_2k
ROOT=$RAW/hermite_distinct
PROC=$ROOT/processed_delay
DD=$PROC/geometry_split
log(){ echo "[2k] $(date +%H:%M:%S) $*"; }

log "STEP 1/3 generate 2000 layouts x 10 traj x 1 theta"
$PY data/generate.py task=random method=hermite \
  channel.signal_type=chirp channel.absolute_delay=true channel.matched_filter=true \
  channel.B=300 channel.df=0.5 channel.signal_n=2048 channel.save_channel_h=false \
  channel.n_traj=10 channel.ppt=50 channel.channel_options=[0.1] \
  hermite.speed_min=2.0 hermite.speed_max=5.0 \
  random_task.layout.aperture_frac=0.20 \
  random_task.n_position_sets=2000 random_task.distinct_trajectories=true \
  random_task.out_dir=$RAW > scratch/2k_gen.log 2>&1
grep -q "^Done\." scratch/2k_gen.log || { log "ABORT: generation failed"; tail -20 scratch/2k_gen.log; exit 1; }
log "generated $(ls -d $ROOT/position_set_* | wc -l) layouts, $(du -sh $RAW | cut -f1)"

log "STEP 2/3 preprocess -> delay features, split 1600/200/200"
$PY data/process_data.py --data-root $ROOT --mode geometry --save-dir $PROC \
   --train-geoms 1600 --val-geoms 200 --test-geoms 200 \
   --delay-features --delay-df 0.5 --delay-roll 50 > scratch/2k_proc.log 2>&1
[ -f "$DD/train_data.pkl" ] || { log "ABORT: preprocessing failed"; tail -20 scratch/2k_proc.log; exit 1; }
log "processed: $(grep -E '^  (train|val  |test )' scratch/2k_proc.log | tr '\n' ' ')"

log "STEP 3/3 training"
C="data=calib_bb data.data_dir=$DD model=spatial_cnp model.spatial.norm_acoustic=none \
   training.batch_size=128 training.num_workers=4 seed=2 wandb.enabled=false"
run(){ local n=$1; shift; TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C "$@" \
        exp_name="$n" > "scratch/${n}.log" 2>&1 & }
# the gate, at full prior size
run PFN_2k            model.spatial.use_position=false
sleep 25
# same, plus rigid-scene translation augmentation: acoustics unchanged, whole
# scene shifted, so the model CANNOT use absolute coordinates and must anchor
# the frame on the context -- exactly the PFN premise.
run PFN_2k_aug        model.spatial.use_position=false data.trans_aug_m=500
sleep 25
# ceiling; max_wavelength raised past the ~1500 m sensor field (default 1000 aliases)
run ORACLE_2k         model.spatial.use_position=true model.spatial.max_wavelength=4000
sleep 25
# prior-size curve with the REAL model (spec 6.9's highest-value ablation)
run PFN_2k_g100       model.spatial.use_position=false data.max_train_geometries=100
sleep 25
run PFN_2k_g400       model.spatial.use_position=false data.max_train_geometries=400
wait
log "ALL COMPLETE"
echo "================ RESULTS (chance ~481 m, analytic 18.4 m) ================"
for n in PFN_2k_g100 PFN_2k_g400 PFN_2k PFN_2k_aug ORACLE_2k; do
  d=$(ls -d output/hydra/training/*/$n/seed_2 2>/dev/null | tail -1)
  [ -n "$d" ] && [ -f "$d/train_log.csv" ] && awk -F, -v N="$n" \
    'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}; lt=$5}
     END{printf "  %-16s best val=%8.2f @ep%-5s (train %.1f)\n",N,m,e,lt}' "$d/train_log.csv"
done
