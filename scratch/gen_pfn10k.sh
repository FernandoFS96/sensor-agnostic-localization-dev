#!/usr/bin/env bash
# A PROPER PFN TEST SET: layout_ood physics, but layout-RICH instead of
# trajectory-rich.
#
# WHY. Every test of the faithful TabPFN joint readout so far ran at ~140-160
# prior datasets. Our own pool curve (pfn_lab) says that regime cannot work:
#   10 tasks -> KL 22.3 | 100 -> 16.0 | 1000 -> 4.34 | 10000 -> 0.249 | fresh -> 0.247
# i.e. a 10,000-task pool is statistically indistinguishable from true
# prior-fitting, and everything below ~1000 is badly under-resourced. The
# existing random_task set is 200 layouts x 400 trajectories x 2 thetas (897 GB)
# -- exactly backwards for a PFN, which needs many DATASETS, not many points.
#
# data.context.calibration is false for this split, so context is the first k
# points of a trajectory and targets are the rest: each trajectory is a
# self-contained task. 3 trajectories per layout is therefore plenty, and the
# budget goes into layouts instead.
#
# PHYSICS IS MATCHED TO random_task EXACTLY (df=50 -> tau=201, legacy sinusoid,
# no absolute delay, no matched filter, snr 10, 2 thetas, aperture_frac 0.5,
# offset_frac 0.3, shared trajectory ensemble) and master_seed stays 11, so
# layout_seed_for(11, p) reproduces the ORIGINAL layouts for p < 200. A CNP run
# capped at 160 layouts is therefore a genuine replication control against the
# existing 23.50 m result.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
OUT=./data/pfn10k
TOTAL=10000
SHARDS=8
PER=$(( TOTAL / SHARDS ))
mkdir -p scratch/pfn10k

GEN="task=random method=hermite
 channel.df=50.0 channel.signal_type=sinusoid channel.absolute_delay=false
 channel.matched_filter=false channel.save_channel_h=false
 channel.n_traj=3 channel.ppt=50 channel.channel_options=[0.1,0.2] channel.snr=10.0
 channel.master_seed=11
 random_task.layout.aperture_frac=0.5 random_task.layout.offset_frac=0.3
 random_task.distinct_trajectories=false
 random_task.n_position_sets=${TOTAL} random_task.out_dir=${OUT}"

echo "=== GENERATION START $(date '+%F %T') ==="
echo "  ${TOTAL} layouts x 3 traj x 2 thetas, ${SHARDS} shards of ${PER}"
S=$(date +%s)
for i in $(seq 0 $((SHARDS-1))); do
  A=$(( i * PER )); B=$(( A + PER ))
  $PY data/generate.py $GEN random_task.start_set=$A random_task.end_set=$B \
      > scratch/pfn10k/shard_${i}.log 2>&1 &
done
wait
E=$(date +%s)
echo "=== GENERATION DONE $(date '+%F %T') -- $(( (E-S)/60 )) min ==="

N=$(ls -d ${OUT}/hermite_shared/position_set_* 2>/dev/null | wc -l)
echo "  layouts on disk: ${N} / ${TOTAL}"
if [ "$N" -ne "$TOTAL" ]; then
  echo "  !! INCOMPLETE -- not processing."; grep -il "error\|traceback" scratch/pfn10k/shard_*.log; exit 1
fi
du -sh ${OUT} | awk '{print "  raw: "$1}'

# RAW features (tau*n_sensors = 2010), matching the representation that gives
# the 23.50 / 16.5 numbers -- delay features top out near 100 m and would put
# every architecture on the same floor, hiding the effect we are measuring.
echo ""
echo "=== PROCESSING $(date '+%F %T') ==="
$PY -u data/process_data.py --mode geometry \
    --data-root ${OUT}/hermite_shared --method hermite_shared \
    --save-dir ${OUT}/hermite_shared/processed \
    --train-geoms 8000 --val-geoms 1000 --test-geoms 1000 \
    > scratch/pfn10k/process.log 2>&1
echo "  rc=$?"
tail -6 scratch/pfn10k/process.log
du -sh ${OUT}/hermite_shared/processed 2>/dev/null | awk '{print "  processed: "$1}'
echo ""
echo "=== ALL COMPLETE $(date '+%F %T') ==="
