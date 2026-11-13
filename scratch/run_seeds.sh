#!/usr/bin/env bash
# SEED TOP-UP: take the two single-seed families to n=3.
#
# Path C and Path A already have 3 seeds. Path D and the clean layout set -- the
# two most recent and most interesting results -- have 1, which leaves their
# close orderings unsupportable: PFN vs RCNP on Path D (25.41 vs 24.39) and
# CNP vs CNP+FFN on the clean set are both inside plausible seed noise at n=1.
# The large gaps are safe either way; these specific claims are not.
#
# Path D goes first: it is the newest result AND the cheaper data (7.7 GB/run
# vs 19.3 GB), so the headline table becomes defensible soonest.
# Configs are copied verbatim from the original runs -- only `seed` differs.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD_D=data/pathD/hermite_distinct/processed/geometry_split
DD_C=data/pfn10kd/hermite_distinct/processed/geometry_split
LOG=scratch/seeds; mkdir -p $LOG

BASE="data.max_train_geometries=null data.max_train_trajectories=null
      data.max_val_trajectories=null wandb.enabled=false"
PSF="data.context.per_sample=false"
CAL="data.context.calibration=true"

run () {   # $1=exp $2=model $3=extra $4=seed $5=data_dir
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE data.data_dir=$5 \
      model=$2 $3 seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

MAXJOBS=4          # ~9 GB resident per Path D run
echo "=== PATH D seeds 1,2 $(date '+%F %T') ==="
for S in 1 2; do
  run PD_A_cnp   spatial_cnp  "$PSF"                       $S $DD_D
  run PD_A_cnpF  spatial_cnp  "$PSF model.attn_ffn=true"   $S $DD_D
  run PD_A_pfn   spatial_pfn  "$PSF model.readout=joint"   $S $DD_D
  run PD_A_rcnp  spatial_rcnp "$PSF"                       $S $DD_D
  run PD_B_cnp   spatial_cnp  "$CAL"                       $S $DD_D
  run PD_B_cnpF  spatial_cnp  "$CAL model.attn_ffn=true"   $S $DD_D
  run PD_B_pfn   spatial_pfn  "$CAL model.readout=joint"   $S $DD_D
done
wait
echo "=== PATH D SEEDS COMPLETE $(date '+%F %T') ==="

MAXJOBS=3          # ~22 GB resident per clean-set run
echo "=== CLEAN LAYOUT SET seeds 1,2 $(date '+%F %T') ==="
for S in 1 2; do
  run CL_A_cnp   spatial_cnp  "$PSF"                       $S $DD_C
  run CL_A_cnpF  spatial_cnp  "$PSF model.attn_ffn=true"   $S $DD_C
  run CL_A_pfn   spatial_pfn  "$PSF model.readout=joint"   $S $DD_C
  run CL_A_rcnp  spatial_rcnp "$PSF"                       $S $DD_C
  run CL_B_cnp   spatial_cnp  "$CAL"                       $S $DD_C
  run CL_B_cnpF  spatial_cnp  "$CAL model.attn_ffn=true"   $S $DD_C
  run CL_B_pfn   spatial_pfn  "$CAL model.readout=joint"   $S $DD_C
done
wait
echo "=== ALL SEEDS COMPLETE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, numpy as np
def cell(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return v
for title,rows in (("PATH D -- held-out environments",
                    [("CNP","PD_A_cnp","PD_B_cnp"),("CNP + FFN","PD_A_cnpF","PD_B_cnpF"),
                     ("PFN (joint)","PD_A_pfn","PD_B_pfn"),("RCNP","PD_A_rcnp",None)]),
                   ("CLEAN LAYOUT SET -- held-out layouts",
                    [("CNP","CL_A_cnp","CL_B_cnp"),("CNP + FFN","CL_A_cnpF","CL_B_cnpF"),
                     ("PFN (joint)","CL_A_pfn","CL_B_pfn"),("RCNP","CL_A_rcnp",None)])):
    print(f"\n  {title}  (mean +/- sd over seeds)\n")
    print(f"  {'model':<18}{'A: prefix':>18}{'B: CALIBRATION':>20}")
    for lbl,ta,tb in rows:
        a,b=cell(ta),cell(tb) if tb else []
        fa=f"{np.mean(a):.2f} +/- {np.std(a):.2f} (n={len(a)})" if a else "-"
        fb=f"{np.mean(b):.2f} +/- {np.std(b):.2f} (n={len(b)})" if b else "n/a (method)"
        print(f"  {lbl:<18}{fa:>18}{fb:>20}")
EOF
