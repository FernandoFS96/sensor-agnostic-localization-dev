#!/usr/bin/env bash
# POSITION-AWARE REFERENCE for the Path C curve.
#
# Path C established that position-FREE localization saturates near 92.8 m. The
# question that makes that number mean something is: what is knowing the sensor
# coordinates actually worth, on the SAME data and the SAME architecture family?
#
# "Add positions" is not a single-variable change -- the position-free arm also
# ran norm_acoustic=none and tokenize=false -- so both readings are run:
#
#   ARM A (PA_)  full spatial_cnp defaults: tokenize=true, use_position=true,
#                norm_acoustic=standardize. The real position-aware model, i.e.
#                the practical reference number for the thesis.
#   ARM C (PAC_) one-variable control: positions ON, everything else IDENTICAL
#                to the position-free arm (tokenize=false, norm_acoustic=none).
#                Isolates the position information itself from the architecture
#                built to exploit it.
#
# Reuses the pre-subset cell dirs, so resident memory stays proportional to each
# cell (the fix that kept the Path C grid off the ~83 GB projection).
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
CELLS=data/calib_10k/cells
LOG=scratch/posaware; mkdir -p $LOG
MAXJOBS=6

BASE="data=calib_bb model=spatial_cnp data.max_val_trajectories=10
      training.batch_size=128 training.num_workers=4 wandb.enabled=false"

launch () {   # $1=layouts $2=traj $3=seed $4=arm(A|C)
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
  if [ "$4" = "C" ]; then
    EXTRA="model.spatial.norm_acoustic=none model.spatial.use_position=true model.spatial.tokenize=false"
    NAME=PAC_g$1_t$2
  else
    EXTRA=""
    NAME=PA_g$1_t$2
  fi
  echo "  -> $NAME s$3  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE $EXTRA \
      data.data_dir=$CELLS/g$1_t$2 \
      data.max_train_geometries=$1 data.max_train_trajectories=$2 \
      seed=$3 exp_name=$NAME > $LOG/${NAME}_s$3.log 2>&1 &
}

echo "=== LAUNCH $(date '+%F %T') ==="
for S in 1 2 3; do launch 8000 50 $S A; done
for S in 1 2 3; do launch 8000 10 $S A; done
for S in 1 2 3; do launch 8000 10 $S C; done
for S in 1 2 3; do launch 4000 10 $S A; done
for S in 1 2 3; do launch 1600 10 $S A; done
for S in 1 2 3; do launch  400 10 $S A; done
for S in 1 2 3; do launch  100 10 $S A; done
wait
echo "=== ALL COMPLETE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, numpy as np
def cell(pfx,g,t):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{pfx}_g{g}_t{t}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return v
def fmt(v): return f"{np.mean(v):7.1f}" if v else "      -"
print("\n  WHAT ARE SENSOR POSITIONS WORTH?  val MAE [m], 3 seeds  (chance ~481)\n")
print(f"  {'layouts':>8}{'position-FREE':>16}{'position-AWARE':>16}{'gain':>10}{'ratio':>8}")
for g in (100,400,1600,4000,8000):
    pf,pa=cell("PC",g,10),cell("PA",g,10)
    if pf and pa:
        a,b=np.mean(pf),np.mean(pa)
        print(f"  {g:>8}{a:>16.1f}{b:>16.1f}{a-b:>+10.1f}{a/b:>7.1f}x")
print("\n  BEST CELL (8000 layouts x 50 trajectories)")
pf,pa=cell("PC",8000,50),cell("PA",8000,50)
if pf and pa:
    print(f"    position-free {np.mean(pf):6.1f} m   position-aware {np.mean(pa):6.1f} m"
          f"   ratio {np.mean(pf)/np.mean(pa):.1f}x")
print("\n  ONE-VARIABLE CONTROL at 8000x10 (positions ON, arch identical to position-free)")
print(f"    position-free (pos off)      : {fmt(cell('PC',8000,10))} m")
print(f"    + positions only (arm C)     : {fmt(cell('PAC',8000,10))} m")
print(f"    + positions + spatial arch(A): {fmt(cell('PA',8000,10))} m")
print("\n  HOW MANY LAYOUTS DOES POSITION-FREE NEED TO MATCH POSITION-AWARE?")
pa8=cell("PA",8000,10)
if pa8:
    tgt=np.mean(pa8); b=0.228
    print(f"    position-aware at 8000 layouts = {tgt:.1f} m")
    if tgt>0:
        f=(102.8/tgt)**(1/b)
        print(f"    position-free would need ~{8000*f:,.0f} layouts (b={b}, optimistic)")
EOF
