#!/usr/bin/env bash
# Master queue (relaunched 2026-11-05 evening after the session teardown killed the
# sweeps at 15:38). Runs detached. Order on the GPU:
#   1. B2 PFN size sweep (restarted from scratch)
#   2. chirp B2 pilot: PFN with the PD_B_pfnset_fixdrop recipe on pathD_chirp
#   3. B1 PFN size sweep
# The pilot data (CPU) is generated in parallel with step 1. pathD_chirp = pathD with
# the B3 physics (chirp, absolute delay, matched filter, B 300 Hz, df 0.5 Hz) and the
# same layout, environments, trajectories and split; features = 4 delay features/sensor.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
OUT=./data/pathD_chirp
DDC=$OUT/hermite_distinct/processed_delay/geometry_split
L=scratch/acoustic/pilot_chirp
mkdir -p $L
echo "queue start $(date '+%F %T')"
(
  echo "=== GENERATION START $(date '+%F %T') ==="
  GEN="task=random method=hermite
   channel.df=0.5 channel.signal_type=chirp channel.absolute_delay=true
   channel.matched_filter=true channel.B=300 channel.signal_n=2048 channel.save_channel_h=false
   channel.n_traj=6 channel.ppt=50 channel.channel_options=[0.1] channel.snr=10.0
   channel.master_seed=11 channel.nop=4
   random_task.layout.aperture_frac=0.5 random_task.layout.offset_frac=0.3
   random_task.layout.fixed_seed=777 random_task.distinct_trajectories=true
   random_task.environment.enabled=true random_task.n_position_sets=4000 random_task.out_dir=$OUT"
  for i in 0 1 2 3; do
    $PY data/generate.py $GEN random_task.start_set=$((i*1000)) random_task.end_set=$((i*1000+1000)) \
        > $L/shard_$i.log 2>&1 &
  done
  wait
  N=$(ls -d $OUT/hermite_distinct/position_set_* | wc -l)
  echo "=== GENERATION DONE $(date '+%F %T'), sets $N / 4000, raw $(du -sh $OUT | cut -f1) ==="
  [ "$N" -ne 4000 ] && { echo "!! INCOMPLETE"; exit 1; }
  $PY -u data/process_data.py --mode geometry --data-root $OUT/hermite_distinct --method hermite_distinct \
      --save-dir $OUT/hermite_distinct/processed_delay --delay-features --delay-df 0.5 --delay-roll 50 \
      --splits-file data/pathD/hermite_distinct/processed/geometry_split/splits.json > $L/process.log 2>&1
  echo "=== PROCESSING DONE $(date '+%F %T') (exit $?) ==="
  $PY scratch/acoustic/pilot_chirp_classical.py --data-dir $DDC --raw-root $OUT/hermite_distinct \
      --out-dir output/eval_paper/b2_chirp_pilot_classical > $L/classical.log 2>&1
  echo "=== CLASSICAL DONE $(date '+%F %T') (exit $?) ==="; cat $L/classical.log
) > $L/data.log 2>&1 &
GEN=$!
bash scratch/acoustic/run_pfn_size_sweep.sh > scratch/acoustic/run_pfn_size_sweep_out.log 2>&1
echo "B2 sweep finished $(date '+%F %T')"
wait $GEN
if [ -f $DDC/train_data.pkl ]; then
  $PY scripts/train/train_np_geometry.py data.data_dir=$DDC \
      data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
      model=spatial_pfn_set model.readout=joint data.rot_aug=true \
      data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
      training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=PD_B_pfnset_chirp seed=0 \
      > $L/train_s0.log 2>&1
  echo "pilot train done $(date '+%F %T') (exit $?)"
  P=$(ls -d output/hydra/training/*/PD_B_pfnset_chirp/seed_0 | tail -1)
  for k in 0 3 5 7; do
    $PY scripts/eval/eval_calibration.py --data-dir $DDC --models chirp=$P/best.pt --drop $k \
        --out-dir output/eval_paper/b2_chirp_pilot_drop$k > $L/eval_drop$k.log 2>&1
  done
  echo "=== PILOT DONE $(date '+%F %T') ==="
else
  echo "!! pilot data missing, pilot skipped"
fi
bash scratch/acoustic/run_pfn_size_sweep_b1.sh > scratch/acoustic/run_pfn_size_sweep_b1_out.log 2>&1
echo "=== QUEUE DONE $(date '+%F %T') ==="
