#!/usr/bin/env bash
# PATH D: can a model infer an UNSEEN acoustic environment from context?
#
# The dataset holds the array PINNED and the trajectories DISTINCT per set, so
# the environment (depth 30-120 m, c 1470-1540, bottom fraction 0.70-0.95,
# spreading 1.4-2.0) is the ONLY thing that differs between tasks, and no
# environment is shared across train/val/test. Separability of the axis is 0.97
# against theta's 0.28, i.e. two environments now differ about as much as two
# random layouts did -- theta could never have carried this task.
#
# REGIME A (per_sample=false): prefix context. Includes RCNP so the recurrent
#   family is represented; on the layout task it was a path extrapolator
#   (1.9x acoustic reliance), and this says whether that repeats here.
# REGIME B (calibration=true): context from a DIFFERENT transit of the SAME
#   environment -- the operationally faithful setting, and the one that proved
#   most context-robust on the layout task. Whitelisted to cnp/pfn by the trainer.
#
# The ablation afterwards is the part that decides whether any number here means
# anything: a model that scores well with the acoustics destroyed is not
# inferring an environment.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pathD/hermite_distinct/processed/geometry_split
LOG=scratch/pathD_runs; mkdir -p $LOG
MAXJOBS=4      # ~9 GB resident per run (7.7 GB train + 1 GB val) of 83 GB free

BASE="data.data_dir=$DD data.max_train_geometries=null
      data.max_train_trajectories=null data.max_val_trajectories=null
      wandb.enabled=false"

run () {   # $1=exp $2=model $3=extra
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 $3 seed=0 exp_name=$1 > $LOG/$1.log 2>&1 &
}

PSF="data.context.per_sample=false"
CAL="data.context.calibration=true"
echo "=== PATH D START $(date '+%F %T') ==="
echo "--- REGIME A: prefix context ---"
run PD_A_cnp   spatial_cnp  "$PSF"
run PD_A_cnpF  spatial_cnp  "$PSF model.attn_ffn=true"
run PD_A_pfn   spatial_pfn  "$PSF model.readout=joint"
run PD_A_rcnp  spatial_rcnp "$PSF"
wait
echo "--- REGIME B: CALIBRATION context ---"
run PD_B_cnp   spatial_cnp  "$CAL"
run PD_B_cnpF  spatial_cnp  "$CAL model.attn_ffn=true"
run PD_B_pfn   spatial_pfn  "$CAL model.readout=joint"
wait
echo "=== TRAINING COMPLETE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, numpy as np
def best(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append((min(float(x["val_mae"]) for x in r),
                            min(float(x["train_mae"]) for x in r)))
        except Exception: pass
    return v[0] if v else None
print("\n  PATH D -- held-out ENVIRONMENTS, val MAE [m]\n")
print(f"  {'model':<22}{'A: prefix':>12}{'B: CALIBRATION':>17}{'A train':>10}")
for lbl,ta,tb in (("CNP","PD_A_cnp","PD_B_cnp"),("CNP + FFN","PD_A_cnpF","PD_B_cnpF"),
                  ("PFN (joint)","PD_A_pfn","PD_B_pfn"),("RCNP (recurrent)","PD_A_rcnp",None)):
    a=best(ta); b=best(tb) if tb else None
    print(f"  {lbl:<22}{(f'{a[0]:.2f}' if a else '-'):>12}"
          f"{(f'{b[0]:.2f}' if b else 'n/a (method)'):>17}{(f'{a[1]:.2f}' if a else '-'):>10}")
EOF

echo ""
echo "=== ACOUSTIC-DEPENDENCE ABLATION $(date '+%F %T') ==="
$PY scratch/ablate_acoustics.py --data $DD --batches 12 \
   --tags PD_A_pfn:pfn,PD_A_cnpF:cnp,PD_A_cnp:cnp,PD_A_rcnp:rcnp 2>&1 | tail -10
echo "=== ALL COMPLETE $(date '+%F %T') ==="
