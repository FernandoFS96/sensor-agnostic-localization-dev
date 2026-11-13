#!/usr/bin/env bash
# PATH C, attempt 2 -- preprocess (restartable) -> per-cell data -> training grid.
#
# Changes from attempt 1, both forced by what went wrong:
#   * preprocessing is chunked + resumable and RETRIED up to 3 times. The first
#     attempt was killed ~1 h in by something I could not attribute (no OOM, no
#     traceback, 941 GB free) and lost everything. A repeat now costs one chunk.
#   * training reads PRE-SUBSET cell directories. TrajectoryDataset pickle.loads
#     the whole train file and the trainer caps only afterwards, so the grid as
#     originally written projected to ~83 GB resident against 116 GB available
#     (measured 10.4 KB/sample x 400k samples x 7 concurrent x num_workers=4).
#
# num_workers stays at 4: it feeds the per-worker context-size RNG, so changing
# it would change the training distribution and break comparability with the
# 463/322/180 baseline this experiment is anchored to.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
SPLIT=data/calib_10k/hermite_distinct/processed_delay/geometry_split
CELLS=data/calib_10k/cells
LOG=scratch/pathC; mkdir -p $LOG
MAXJOBS=6

# ---- 1. preprocessing, with resume ------------------------------------------
for attempt in 1 2 3; do
  if [ -f "$SPLIT/train_data.pkl" ] && [ -f "$SPLIT/metadata.pkl" ]; then break; fi
  echo "=== PREPROCESS attempt $attempt  $(date '+%F %T') ==="
  $PY -u scratch/preprocess_10k.py >> $LOG/preprocess.log 2>&1
  echo "    rc=$? $(date '+%T')"
done
if [ ! -f "$SPLIT/train_data.pkl" ]; then
  echo "!! preprocessing failed 3x -- aborting."; tail -30 $LOG/preprocess.log; exit 1
fi

# ---- 2. integrity gate -------------------------------------------------------
$PY - <<'EOF' || { echo "!! integrity FAILED -- nothing launched."; exit 1; }
import pickle, numpy as np, sys
d="data/calib_10k/hermite_distinct/processed_delay/geometry_split"
seen={}
for s_ in ("train","val","test"):
    s=pickle.load(open(f"{d}/{s_}_data.pkl","rb"))
    g={int(r["geometry_id"]) for r in s}; seen[s_]=g
    X=np.asarray(s[0]["X"])
    print(f"  {s_:<6}: {len(s):>8,} samples  X={X.shape} {X.dtype}  {len(g):>5} geoms")
    assert np.isfinite(np.stack([r["y"] for r in s[:200]])).all(), f"non-finite y in {s_}"
for a,b in (("train","val"),("train","test"),("val","test")):
    ov=seen[a]&seen[b]
    assert not ov, f"{len(ov)} geometries shared by {a}/{b} -- layout leak"
print("  integrity OK: no geometry shared across splits")
EOF

# ---- 3. per-cell data dirs ---------------------------------------------------
if [ ! -d "$CELLS/g100_t10" ]; then
  echo "=== BUILD CELLS $(date '+%T') ==="
  $PY -u scratch/make_cells.py >> $LOG/cells.log 2>&1 || { echo "!! make_cells failed"; tail -20 $LOG/cells.log; exit 1; }
  grep -E "samples|shared val" $LOG/cells.log | tail -9
fi

# ---- 4. grid -----------------------------------------------------------------
BASE="data=calib_bb model=spatial_cnp model.spatial.norm_acoustic=none
      model.spatial.use_position=false model.spatial.tokenize=false
      data.max_val_trajectories=10 training.batch_size=128
      training.num_workers=4 wandb.enabled=false"

launch () {   # $1=layouts $2=traj $3=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
  echo "  -> g$1 t$2 s$3  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      data.data_dir=$CELLS/g$1_t$2 \
      data.max_train_geometries=$1 data.max_train_trajectories=$2 \
      seed=$3 exp_name=PC_g$1_t$2 > $LOG/g$1_t$2_s$3.log 2>&1 &
}

echo "=== LAUNCH GRID $(date '+%F %T') ==="
for S in 1 2 3; do launch 8000 50 $S; done
for S in 1 2 3; do launch 8000 10 $S; done
for S in 1 2 3; do launch 4000 10 $S; done
for S in 1 2 3; do launch 1600 50 $S; done
for S in 1 2 3; do launch 1600 10 $S; done
for S in 1 2 3; do launch  400 10 $S; done
for S in 1 2 3; do launch  100 10 $S; done
wait
echo "=== ALL RUNS COMPLETE $(date '+%F %T') ==="

# ---- 5. results --------------------------------------------------------------
$PY - <<'EOF'
import csv, glob, numpy as np
def cell(g,t):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/PC_g{g}_t{t}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return v
print("\n  PATH C -- position-free CALIBRATION, val MAE [m]  (chance ~481)")
print("\n  MAIN CURVE (trajectories pinned at 10 -> layouts vary alone)")
print(f"  {'layouts':>8}{'mean':>9}{'sd':>7}{'seeds':>24}{'old 2k set':>12}")
old={100:463,400:322,1600:180}
for g in (100,400,1600,4000,8000):
    v=cell(g,10)
    if v: print(f"  {g:>8}{np.mean(v):>9.1f}{(np.std(v) if len(v)>1 else 0):>7.1f}"
                f"{str([round(x,1) for x in v]):>24}{old.get(g,'-'):>12}")
print("\n  TRAJECTORY AXIS (same layouts, 10 vs 50 trajectories)")
print(f"  {'layouts':>8}{'t=10':>10}{'t=50':>10}{'delta':>9}")
for g in (1600,8000):
    a,b=cell(g,10),cell(g,50)
    if a and b: print(f"  {g:>8}{np.mean(a):>10.1f}{np.mean(b):>10.1f}{np.mean(b)-np.mean(a):>+9.1f}")
EOF
