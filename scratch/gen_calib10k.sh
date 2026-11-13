#!/usr/bin/env bash
# Path C dataset: 10,000 layouts x 50 trajectories, broadband/matched-filter.
#
# Sharded 8 ways over DISJOINT position-set ranges. Layout and trajectory seeds
# are both derived from the GLOBAL set index p, so this reproduces a serial run
# set-for-set (verified: trajectories / sensor_positions / channel_info are
# bit-identical; only the additive measurement-noise realisation differs, and
# that is already non-reproducible run-to-run in the serial path because
# filter() draws its noise inside multiprocessing workers).
#
# 8 shards, not 16: filter() already parallelises internally, so throughput
# plateaus at ~3.2x (measured 0.139 s/traj at 8-way vs 0.141 at 16-way).
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
OUT=./data/calib_10k
TOTAL=10000
SHARDS=8
PER=$(( TOTAL / SHARDS ))
mkdir -p scratch/gen10k

GEN="task=random method=hermite
 channel.signal_type=chirp channel.absolute_delay=true channel.matched_filter=true
 channel.B=300 channel.df=0.5 channel.signal_n=2048 channel.save_channel_h=false
 channel.n_traj=50 channel.ppt=50 channel.channel_options=[0.1]
 hermite.speed_min=2.0 hermite.speed_max=5.0
 random_task.layout.aperture_frac=0.20 random_task.distinct_trajectories=true
 random_task.n_position_sets=${TOTAL} random_task.out_dir=${OUT}"

echo "=== GENERATION START $(date '+%F %T') ==="
echo "  ${TOTAL} layouts x 50 traj, ${SHARDS} shards of ${PER}"
S=$(date +%s)
for i in $(seq 0 $((SHARDS-1))); do
  A=$(( i * PER )); B=$(( A + PER ))
  $PY data/generate.py $GEN random_task.start_set=$A random_task.end_set=$B \
      > scratch/gen10k/shard_${i}.log 2>&1 &
done
wait
E=$(date +%s)
echo "=== GENERATION DONE $(date '+%F %T') -- $(( (E-S)/60 )) min ==="

NSETS=$(ls -d ${OUT}/hermite_distinct/position_set_* 2>/dev/null | wc -l)
echo "  layouts on disk: ${NSETS} / ${TOTAL}"
if [ "$NSETS" -ne "$TOTAL" ]; then
  echo "  !! INCOMPLETE -- not processing. Check scratch/gen10k/shard_*.log"
  grep -il "error\|traceback" scratch/gen10k/shard_*.log
  exit 1
fi
du -sh ${OUT} | awk '{print "  raw size: "$1}'

# Delay features: 4 per sensor, the representation that gave the 180 m result.
# Geometry split is computed over ALL 10k layouts here, so any later subsample
# of layouts inherits a split that never mixes a geometry across train/val/test.
echo ""
echo "=== PROCESSING $(date '+%F %T') ==="
$PY data/process_data.py --mode geometry \
    --data-root ${OUT}/hermite_distinct --method hermite_distinct \
    --save-dir ${OUT}/hermite_distinct/processed_delay \
    --delay-features --delay-df 0.5 --delay-roll 50 \
    --train-geoms 8000 --val-geoms 1000 --test-geoms 1000 \
    > scratch/gen10k/process.log 2>&1
RC=$?
echo "  process rc=${RC}"
tail -5 scratch/gen10k/process.log
du -sh ${OUT}/hermite_distinct/processed_delay 2>/dev/null | awk '{print "  features: "$1}'
echo ""
echo "=== ALL COMPLETE $(date '+%F %T') ==="
