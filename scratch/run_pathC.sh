#!/usr/bin/env bash
# PATH C -- does the position-free arm keep improving with more layouts?
#
# The prior curve (463 / 322 / 180 at 100 / 400 / 1600 layouts, Screen A) was
# steep and unsaturated. This extends it to 8000 on the new 10k-layout set.
#
# Two controls are built in:
#   * REPLICATION -- 100/400/1600 are re-run on the NEW data. They should land
#     near 463/322/180. Exact equality is not expected (different layouts,
#     different noise draw, different val set), but a large disagreement would
#     mean the new dataset is not the same experiment and every new number below
#     it would be suspect.
#   * AXIS SEPARATION -- the old data had 10 trajectories per layout, the new
#     one has 50. The main curve therefore PINS trajectories at 10 so it varies
#     layouts alone; a separate arm varies trajectories at fixed layouts to
#     price that axis by itself.
#
# max_val_trajectories is pinned at 10 in EVERY cell, so the validation metric
# is computed on an identical 10,000-sample set throughout and MAE is comparable
# across cells rather than drifting with the training config.
#
# Architecture is the one Screen A proved works for the position-free task:
# spatial_cnp with use_position=false (no sensor coordinates) and
# tokenize=false (the flat readout; the tokenised one is structurally broken
# here -- it was what held the old curve at 427 instead of 180).
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/calib_10k/hermite_distinct/processed_delay/geometry_split
LOG=scratch/pathC
mkdir -p $LOG
MAXJOBS=7          # peak cell is 8000x50 ~ 4 GB resident; 7 keeps us well inside RAM

# ---- 1. wait for the preprocessing chain, then verify the data is real -------
while ps -p 905399 >/dev/null 2>&1; do sleep 60; done
echo "=== preprocessing finished $(date '+%F %T') ==="
if [ ! -d "$DD" ] || [ -z "$(ls -A $DD 2>/dev/null)" ]; then
  echo "!! $DD missing or empty -- aborting, nothing launched."
  tail -20 scratch/gen10k/process.log; exit 1
fi
$PY - <<'EOF' || exit 1
import pickle, os, sys, numpy as np
d = "data/calib_10k/hermite_distinct/processed_delay/geometry_split"
ok, seen = True, {}
for split in ("train", "val", "test"):
    p = f"{d}/{split}_data.pkl"
    if not os.path.exists(p):
        print(f"  !! missing {p}"); ok = False; continue
    s = pickle.load(open(p, "rb"))
    g = {int(r["geometry_id"]) for r in s}
    t = {int(r["traj_id"]) for r in s}
    X = np.asarray(s[0]["X"])
    seen[split] = g
    print(f"  {split:<6}: {len(s):>8,} samples | X={X.shape} | "
          f"{len(g):>5} geometries x {len(t):>3} traj")
    if not np.isfinite(np.asarray(s[0]["y"], dtype=float)).all():
        print(f"  !! non-finite y in {split}"); ok = False
# A geometry appearing in two splits would leak the layout being generalised to.
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    if a in seen and b in seen and seen[a] & seen[b]:
        print(f"  !! {len(seen[a] & seen[b])} geometries shared by {a}/{b}"); ok = False
print("  split integrity:", "OK -- no geometry shared across splits" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
echo ""

# ---- 2. the grid --------------------------------------------------------------
BASE="data=calib_bb data.data_dir=$DD model=spatial_cnp
      model.spatial.norm_acoustic=none model.spatial.use_position=false
      model.spatial.tokenize=false data.max_val_trajectories=10
      training.batch_size=128 training.num_workers=4 wandb.enabled=false"

launch () {   # $1=layouts $2=traj $3=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
  echo "  -> g$1 t$2 s$3  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      data.max_train_geometries=$1 data.max_train_trajectories=$2 \
      seed=$3 exp_name=PC_g$1_t$2 > $LOG/g$1_t$2_s$3.log 2>&1 &
}

echo "=== LAUNCHING $(date '+%F %T') ==="
# Big cells first so the long pole starts earliest and the tail is short.
for S in 1 2 3; do launch 8000 50 $S; done
for S in 1 2 3; do launch 8000 10 $S; done
for S in 1 2 3; do launch 4000 10 $S; done
for S in 1 2 3; do launch 1600 50 $S; done
for S in 1 2 3; do launch 1600 10 $S; done
for S in 1 2 3; do launch  400 10 $S; done
for S in 1 2 3; do launch  100 10 $S; done
wait
echo "=== ALL RUNS COMPLETE $(date '+%F %T') ==="

# ---- 3. results ---------------------------------------------------------------
$PY - <<'EOF'
import csv, glob, numpy as np
def cell(g, t):
    v = []
    for d in glob.glob(f"output/hydra/training/*/PC_g{g}_t{t}/seed_*"):
        try:
            r = [x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return v
print("\n  PATH C -- position-free CALIBRATION, val MAE [m]   (chance ~481)")
print("\n  MAIN CURVE (trajectories pinned at 10, so layouts vary alone)")
print(f"  {'layouts':>8}{'mean':>9}{'sd':>7}{'seeds':>22}   {'old (2k set)':>12}")
old = {100: 463, 400: 322, 1600: 180}
for g in (100, 400, 1600, 4000, 8000):
    v = cell(g, 10)
    if v:
        o = f"{old[g]}" if g in old else "-"
        print(f"  {g:>8}{np.mean(v):>9.1f}{(np.std(v) if len(v)>1 else 0):>7.1f}"
              f"{str([round(x,1) for x in v]):>22}   {o:>12}")
print("\n  TRAJECTORY AXIS (same layouts, 10 vs 50 trajectories each)")
print(f"  {'layouts':>8}{'t=10':>10}{'t=50':>10}{'delta':>9}")
for g in (1600, 8000):
    a, b = cell(g, 10), cell(g, 50)
    if a and b:
        print(f"  {g:>8}{np.mean(a):>10.1f}{np.mean(b):>10.1f}{np.mean(b)-np.mean(a):>+9.1f}")
EOF
