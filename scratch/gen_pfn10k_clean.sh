#!/usr/bin/env bash
# CLEAN rebuild of the PFN test set: distinct_trajectories=TRUE.
#
# WHAT WENT WRONG. The first build used distinct_trajectories=false to match
# random_task, but random_task had 400 trajectories per layout where this set has
# 3 -- so "shared" collapsed the whole dataset to THREE distinct paths, reused
# byte-identically across all 10,000 layouts. Verified: every sample with a given
# traj_id has an identical path. A temporal model then needs only to recognise
# which of three paths it is on (2 context points suffice) and recall the rest,
# which is exactly what the ablation caught: RANP scored 1.32 m intact and still
# 3.09 m with the acoustics replaced by pure NOISE.
#
# distinct_trajectories=true gives every (layout, theta) its OWN trajectories,
# seeded by traj_seed_for(master_seed, p, offset), so there is no finite path set
# to memorise. Everything else is unchanged from the first build (df=50 -> tau=201,
# legacy sinusoid, snr 10, 2 thetas, aperture 0.5 / offset 0.3, master_seed 11),
# so the two sets differ in exactly one property.
#
# n_traj=3 is kept: it matches the previous size (48,000 train samples, ~19 GB)
# AND satisfies the CALIBRATION regime's requirement of >=2 trajectories per
# (geometry, theta) for drawing a separate calibration transit.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
OUT=./data/pfn10kd
TOTAL=10000
SHARDS=8
PER=$(( TOTAL / SHARDS ))
mkdir -p scratch/pfn10kd

GEN="task=random method=hermite
 channel.df=50.0 channel.signal_type=sinusoid channel.absolute_delay=false
 channel.matched_filter=false channel.save_channel_h=false
 channel.n_traj=3 channel.ppt=50 channel.channel_options=[0.1,0.2] channel.snr=10.0
 channel.master_seed=11
 random_task.layout.aperture_frac=0.5 random_task.layout.offset_frac=0.3
 random_task.distinct_trajectories=true
 random_task.n_position_sets=${TOTAL} random_task.out_dir=${OUT}"

echo "=== GENERATION START $(date '+%F %T') ==="
S=$(date +%s)
for i in $(seq 0 $((SHARDS-1))); do
  A=$(( i * PER )); B=$(( A + PER ))
  $PY data/generate.py $GEN random_task.start_set=$A random_task.end_set=$B \
      > scratch/pfn10kd/shard_${i}.log 2>&1 &
done
wait
E=$(date +%s)
echo "=== GENERATION DONE $(date '+%F %T') -- $(( (E-S)/60 )) min ==="

N=$(ls -d ${OUT}/hermite_distinct/position_set_* 2>/dev/null | wc -l)
echo "  layouts on disk: ${N} / ${TOTAL}"
[ "$N" -ne "$TOTAL" ] && { echo "  !! INCOMPLETE"; grep -il "error\|traceback" scratch/pfn10kd/shard_*.log; exit 1; }
du -sh ${OUT} | awk '{print "  raw: "$1}'

echo ""
echo "=== PROCESSING $(date '+%F %T') ==="
$PY -u data/process_data.py --mode geometry \
    --data-root ${OUT}/hermite_distinct --method hermite_distinct \
    --save-dir ${OUT}/hermite_distinct/processed \
    --train-geoms 8000 --val-geoms 1000 --test-geoms 1000 \
    > scratch/pfn10kd/process.log 2>&1
echo "  rc=$?"; tail -5 scratch/pfn10kd/process.log

# ---- the check the first build FAILED -----------------------------------------
$PY - <<'EOF'
import pickle, numpy as np
d="data/pfn10kd/hermite_distinct/processed/geometry_split"
s=pickle.load(open(f"{d}/val_data.pkl","rb"))
by={}
for r in s[:900]:
    by.setdefault(int(r["traj_id"]),[]).append(np.asarray(r["y"]))
print("\n  TRAJECTORY-UNIQUENESS CHECK (this is what the old set failed)")
bad=0
for tid,l in sorted(by.items()):
    a=l[0]; same=sum(1 for y in l[1:40] if np.allclose(y,a,atol=1e-6))
    bad+=same
    print(f"    traj_id={tid}: {same}/{min(39,len(l)-1)} layouts share an IDENTICAL path")
print("  VERDICT:", "CLEAN -- paths are per-layout unique" if bad==0
      else f"STILL SHARED ({bad} duplicates)")
EOF
echo ""
echo "=== ALL COMPLETE $(date '+%F %T') ==="
