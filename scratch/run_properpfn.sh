#!/usr/bin/env bash
# THE PROPER PFN TEST -- does the faithful TabPFN readout work once the prior
# pool is large enough?
#
# Every previous test of readout=joint ran at ~140-160 prior datasets. The
# pfn_lab pool curve says that regime cannot work:
#   10 -> KL 22.3 | 100 -> 16.0 | 1000 -> 4.34 | 10000 -> 0.249 | fresh -> 0.247
# so "the joint readout is broken" was measured exactly where failure was
# guaranteed. This runs it at 8000 layouts, 50x more, on data built for it.
#
# THREE ARMS, identical data / protocol / step count:
#   spatial_cnp        the original baseline architecture (0.96M)
#   spatial_pfn cross  transformer backbone, CNP-style readout (1.99M)
#   spatial_pfn joint  the FAITHFUL PFN: label in the token, targets attend
#                      context only, one unified stack (1.58M)
#
# Note joint is SMALLER than cross (1.58M vs 1.99M) -- it has no separate cross
# blocks -- so if joint wins it does so from a capacity disadvantage, and if it
# loses, capacity is a confound to state honestly rather than a rescue.
#
# The decisive cell is joint@8000. Reading against cross@8000 answers whether
# prior-fitting is viable for this problem, and against cnp@8000 whether it beats
# what you already had.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pfn10k/hermite_shared/processed/geometry_split
LOG=scratch/properpfn; mkdir -p $LOG
MAXJOBS=3      # ~22 GB resident per run (19.3 GB train + 2.4 GB val) of 121 GB

# Caps are explicitly null: this dataset is layout-rich by construction and the
# whole point is to use every layout.
BASE="data.data_dir=$DD data.max_train_geometries=null
      data.max_train_trajectories=null data.max_val_trajectories=null
      wandb.enabled=false"

run () {   # $1=exp_name $2=model $3=extra $4=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 $3 seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

echo "=== PROPER PFN TEST START $(date '+%F %T') ==="
for S in 0 1; do
  run PT_joint spatial_pfn "model.readout=joint" $S
  run PT_cross spatial_pfn "model.readout=cross" $S
  run PT_cnp   spatial_cnp ""                    $S
done
wait
echo "=== COMPLETE $(date '+%F %T') ==="

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
    return v
print("\n  PROPER PFN TEST -- 8000 prior layouts, raw 2010-dim features\n")
print(f"  {'arm':<34}{'params':>8}{'val MAE':>10}{'train MAE':>11}   seeds")
for tag,lbl,p in (("PT_cnp","spatial_CNP (baseline)","0.96M"),
                  ("PT_cross","spatial_PFN cross readout","1.99M"),
                  ("PT_joint","spatial_PFN JOINT (true PFN)","1.58M")):
    v=best(tag)
    if v:
        print(f"  {lbl:<34}{p:>8}{np.mean([x[0] for x in v]):>10.2f}"
              f"{np.mean([x[1] for x in v]):>11.2f}   {[round(x[0],2) for x in v]}")
print("\n  for reference, the SAME architectures on the 160-layout layout_ood set:")
print(f"    spatial_CNP 23.50 | spatial_PFN cross 16.46 | joint never tested at scale")
EOF
