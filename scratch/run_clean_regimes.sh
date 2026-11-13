#!/usr/bin/env bash
# Two regimes on the CLEAN (per-layout-unique trajectory) set.
#
# REGIME A -- calibration=false, the original setting. Context is an ordered
#   PREFIX of the target's own trajectory, so path continuation is still
#   available; what is gone is the ability to MEMORISE a 3-path set. Includes the
#   recurrent models, which is the point: RANP scored 1.32 m on the contaminated
#   set and 3.09 m with the acoustics replaced by noise. If it stays far ahead
#   here it was exploiting continuation, not memorisation; if it collapses, the
#   old ranking was memorisation all along.
#
# REGIME B -- calibration=true. Context comes from a DIFFERENT trajectory of the
#   same (geometry, theta), so there is no prefix of the target path to continue
#   and the acoustics are the only route to the answer. This is the operationally
#   faithful setting: a calibration transit is never the track you want to
#   localize. NOTE the trainer whitelists calibration to cnp/pfn -- the recurrent
#   models cannot express cross-trajectory context at all, so their absence here
#   is a property of the method, not an omission.
#
# Every arm is followed by the acoustic-dependence ablation, which is the only
# thing that distinguishes a correct answer from a lucky one.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pfn10kd/hermite_distinct/processed/geometry_split
LOG=scratch/clean; mkdir -p $LOG
MAXJOBS=3

for f in train_data.pkl val_data.pkl test_data.pkl metadata.pkl; do
  while [ ! -s "$DD/$f" ]; do sleep 30; done
done
sleep 20   # metadata.pkl is written last; give the final flush a moment
echo "=== data ready $(date '+%F %T') ==="

BASE="data.data_dir=$DD data.max_train_geometries=null
      data.max_train_trajectories=null data.max_val_trajectories=null
      wandb.enabled=false"

run () {   # $1=exp $2=model $3=extra $4=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 $3 seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

PSF="data.context.per_sample=false"
echo "=== REGIME A (calibration=false) $(date '+%F %T') ==="
run CL_A_ranp spatial_ranp "$PSF"                                       0
run CL_A_rcnp spatial_rcnp "$PSF"                                       0
run CL_A_cnp  spatial_cnp  "$PSF"                                       0
run CL_A_pfn  spatial_pfn  "$PSF model.readout=joint"                   0
run CL_A_cnpF spatial_cnp  "$PSF model.attn_ffn=true"                   0
wait
echo "=== REGIME A COMPLETE $(date '+%F %T') ==="

echo "=== REGIME B (calibration=true; cnp/pfn only by design) $(date '+%F %T') ==="
CAL="data.context.calibration=true"
run CL_B_cnp  spatial_cnp  "$CAL"                                       0
run CL_B_pfn  spatial_pfn  "$CAL model.readout=joint"                   0
run CL_B_cnpF spatial_cnp  "$CAL model.attn_ffn=true"                   0
wait
echo "=== REGIME B COMPLETE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, numpy as np
def best(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return np.mean(v) if v else None
rows=[("RANP (recurrent)","CL_A_ranp",None),("RCNP (recurrent)","CL_A_rcnp",None),
      ("CNP","CL_A_cnp","CL_B_cnp"),("CNP + FFN","CL_A_cnpF","CL_B_cnpF"),
      ("PFN (joint)","CL_A_pfn","CL_B_pfn")]
print("\n  CLEAN SET (per-layout-unique trajectories) -- val MAE [m]\n")
print(f"  {'model':<22}{'A: prefix ctx':>16}{'B: CALIBRATION':>17}")
for lbl,ta,tb in rows:
    a=best(ta); b=best(tb) if tb else None
    print(f"  {lbl:<22}{(f'{a:.2f}' if a else '-'):>16}"
          f"{(f'{b:.2f}' if b else 'n/a (method)'):>17}")
print("\n  contaminated set for reference: RANP 3.47 RCNP 4.49 PFN 7.99 CNP+FFN 9.61 CNP 11.79")
EOF
echo ""
echo "=== ABLATION ON THE CLEAN MODELS $(date '+%F %T') ==="
$PY scratch/ablate_acoustics.py --data $DD --tags CL_A_ranp:ranp,CL_A_rcnp:rcnp,CL_A_pfn:pfn,CL_A_cnpF:cnp,CL_A_cnp:cnp 2>&1 | tail -14
echo "=== ALL COMPLETE $(date '+%F %T') ==="
