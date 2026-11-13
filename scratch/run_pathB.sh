#!/usr/bin/env bash
# PATH B -- PFN on the layout-OOD task, mimicking the spatial_cnp protocol.
#
# REFERENCE: spatial_cnp_random_hermite_shared_layout_ood_5e-05_aug_m150
#            best val 23.50 m @ep494 (0.93M params, 35.8 s/epoch).
# That run used pure defaults, so this changes ONLY the model and matches
# everything else exactly: lr 5e-05, 1000 epochs, batch 128, num_workers 4,
# trans_aug_m=150, context min2/max40/eval10 per_sample, caps 140 geoms x 350
# trajectories, dataset random_hermite_shared, split layout_ood.
#
# TWO DELIBERATE DEVIATIONS, both forced by prior evidence:
#
#  1. readout=cross, NOT the pfn.yaml default of joint. The TabPFN-faithful joint
#     readout adds the label into the context TOKEN, so query-key matching happens
#     in a mixed acoustics+position space; on a small prior pool it measurably
#     fails to generalise (topology task: train 28 / val 71, against a same-size
#     CNP at 23 / 9). Running joint here would burn 11 h re-deriving that.
#
#  2. A capacity-matched arm. The PFN at pfn.yaml defaults is 1.99M params
#     against the CNP's 0.93M, so a raw win/loss would confound architecture with
#     size. num_hidden=88 gives 0.95M at equal depth -- the honest comparison.
#
# Data is the PRE-CAPPED split: the trainer pickle.loads all 51.5 GB before
# applying the 140/350 caps, which makes concurrent seeds impossible. The capped
# copy holds exactly what those caps would keep (98,000 samples, 39.4 GB), so the
# runs are identical; the caps stay on the command line as no-op assertions.
# MAXJOBS=2 because each run is ~45 GB resident against 120 GB available.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
LOG=scratch/pathB; mkdir -p $LOG
MAXJOBS=2

BASE="model=spatial_pfn model.readout=cross data.data_dir=data/layout_ood_capped
      data.max_train_geometries=140 data.max_train_trajectories=350
      data.max_val_trajectories=350 wandb.enabled=false"

run () {   # $1=exp_name $2=num_hidden $3=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$3 (num_hidden=$2)  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model.num_hidden=$2 seed=$3 exp_name=$1 > $LOG/$1_s$3.log 2>&1 &
}

echo "=== PATH B START $(date '+%F %T') ==="
for S in 0 1 2; do run PFNB_ood_h128 128 $S; done   # default capacity, 3 seeds
run PFNB_ood_h88 88 0                               # capacity-matched to the CNP
wait
echo "=== PATH B COMPLETE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, numpy as np
def best(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return v
print("\n  PATH B -- layout-OOD, val MAE [m]   (reference spatial_cnp = 23.50, 0.93M params)\n")
for tag,lbl,p in (("PFNB_ood_h128","spatial_PFN (1.99M)",1.99),
                  ("PFNB_ood_h88","spatial_PFN capacity-matched (0.95M)",0.95)):
    v=best(tag)
    if v:
        print(f"  {lbl:<40}{np.mean(v):7.2f}  seeds {[round(x,2) for x in v]}")
print(f"  {'spatial_CNP reference (0.93M)':<40}{23.50:7.2f}")
EOF
