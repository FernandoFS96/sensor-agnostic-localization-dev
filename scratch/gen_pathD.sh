#!/usr/bin/env bash
# PATH D dataset: the ENVIRONMENT as the held-out axis.
#
# ENVIRONMENTS-ONLY SPLIT. Three properties must hold at once, and each one is
# there to close a specific failure we have already hit:
#
#   1. environment varies per set      -> the latent the model must infer.
#      theta could not serve: it is one scalar scaling small-scale variability,
#      and measured separability was 0.28 (BELOW ordinary trajectory variation).
#      The sampled environment scores 0.97 on the same measure.
#   2. layout PINNED across all sets   -> otherwise a held-out set differs in
#      both layout and environment and the axes cannot be separated. Pinning
#      needs fixed_positions, not just a fixed seed: the array is fitted to each
#      set's trajectory field extent, so the same seed still moves it.
#   3. trajectories DISTINCT per set   -> this is what pfn10k got wrong. Shared
#      trajectories let a model memorise a finite path set and skip the acoustics
#      entirely (RANP: 3.47 m intact, 3.09 m with acoustics replaced by noise).
#
# Physics matches the layout_ood family (df=50 -> tau=201, legacy sinusoid,
# snr 10) so feature dimensionality stays 2010 and results are comparable.
# ONE theta only: mixing thetas would add a second nuisance axis on top of the
# environment we are trying to isolate.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
OUT=./data/pathD
TOTAL=4000          # 4000 distinct environments
# 4 shards, and nop=4 INSIDE each: channel.nop=-1 makes every shard fork
# joblib workers across all 64 cores, and 8 shards doing that alongside the
# paper sweep drove the box to zero headroom and filled swap. 4x4=16 workers
# is bounded and leaves the sweep untouched.
SHARDS=4
PER=$(( TOTAL / SHARDS ))
mkdir -p scratch/pathD

GEN="task=random method=hermite
 channel.df=50.0 channel.signal_type=sinusoid channel.absolute_delay=false
 channel.matched_filter=false channel.save_channel_h=false
 channel.n_traj=6 channel.ppt=50 channel.channel_options=[0.1] channel.snr=10.0
 channel.master_seed=11 channel.nop=4
 random_task.layout.aperture_frac=0.5 random_task.layout.offset_frac=0.3
 random_task.layout.fixed_seed=777
 random_task.distinct_trajectories=true
 random_task.environment.enabled=true
 random_task.n_position_sets=${TOTAL} random_task.out_dir=${OUT}"

echo "=== GENERATION START $(date '+%F %T') ==="
echo "  ${TOTAL} environments x 6 traj x 1 theta, pinned layout, ${SHARDS} shards"
S=$(date +%s)
for i in $(seq 0 $((SHARDS-1))); do
  A=$(( i * PER )); B=$(( A + PER ))
  $PY data/generate.py $GEN random_task.start_set=$A random_task.end_set=$B \
      > scratch/pathD/shard_${i}.log 2>&1 &
done
wait
echo "=== GENERATION DONE $(date '+%F %T') -- $(( ($(date +%s)-S)/60 )) min ==="

N=$(ls -d ${OUT}/hermite_distinct/position_set_* 2>/dev/null | wc -l)
echo "  sets on disk: ${N} / ${TOTAL}"
[ "$N" -ne "$TOTAL" ] && { echo "  !! INCOMPLETE"; grep -il "error\|traceback" scratch/pathD/shard_*.log; exit 1; }
du -sh ${OUT} | awk '{print "  raw: "$1}'

echo ""
echo "=== PROCESSING $(date '+%F %T') ==="
$PY -u data/process_data.py --mode geometry \
    --data-root ${OUT}/hermite_distinct --method hermite_distinct \
    --save-dir ${OUT}/hermite_distinct/processed \
    --train-geoms 3200 --val-geoms 400 --test-geoms 400 \
    > scratch/pathD/process.log 2>&1
echo "  rc=$?"; tail -5 scratch/pathD/process.log

# ---- the three properties, verified on the produced data ----------------------
$PY - <<'EOF'
import numpy as np, glob, pickle
d="data/pathD/hermite_distinct"
sp=sorted(glob.glob(f"{d}/position_set_*/**/sensor_positions_*.npy",recursive=True))[:200]
a=np.load(sp[0])
print("\n  PATH D DATA CHECKS")
print(f"    layout pinned (x,y identical across {len(sp)} sets): "
      f"{all(np.allclose(np.load(f)[:2],a[:2]) for f in sp[1:])}")
tr=sorted(glob.glob(f"{d}/position_set_*/**/trajectories.npy",recursive=True))[:60]
t0=np.load(tr[0])
print(f"    trajectories distinct per set: {not any(np.allclose(np.load(f),t0) for f in tr[1:])}")
envs={}
for f in sorted(glob.glob(f"{d}/_manifest*.pkl")):
    envs.update(pickle.load(open(f,"rb")).get("environments",{}))
if envs:
    h=[e["h0"] for e in envs.values()]; c=[e["c"] for e in envs.values()]
    print(f"    environments drawn: {len(envs)}  depth {min(h):.0f}-{max(h):.0f} m  "
          f"c {min(c):.0f}-{max(c):.0f} m/s")
s=pickle.load(open(f"{d}/processed/geometry_split/val_data.pkl","rb"))
print(f"    val: {len(s):,} samples, {len({int(r['geometry_id']) for r in s})} environments, "
      f"X={np.asarray(s[0]['X']).shape}, finite={np.isfinite(np.asarray(s[0]['X'])).all()}")
EOF
echo ""
echo "=== ALL COMPLETE $(date '+%F %T') ==="
