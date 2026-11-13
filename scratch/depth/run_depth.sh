#!/usr/bin/env bash
# B2 depth-stratified study: is the PFN's deep-water penalty architectural, or an
# artefact of the prior's depth coverage?
#
# THREE ARMS, and the third already exists:
#   full     -- the PD_A_* runs: 3200 deployments, uniform in depth (350-384 per
#               10 m bin), so a "depth-balanced" arm would be a no-op. Reused.
#   shallow  -- 1613 deployments, h0 in [30, 74.2)
#   deep     -- 1587 deployments, h0 in [74.2, 120]
# Shallow and deep are disjoint, exhaustive and size-matched to 1.6%, so shallow
# is the SIZE control: it isolates "half the data" from "half the depth range".
# Validation and test are untouched in every arm, so the held-out pool is
# identical and only the training distribution moves.
#
# Reading the result on the DEEP test bucket:
#   deep-only ~ full      -> coverage was already sufficient; penalty architectural
#   deep-only < full      -> the breadth of the prior was diluting deep water
#   deep-only > full AND shallow-only > full by the same factor -> pure size effect,
#                            says nothing about depth
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
LOG=scratch/depth; mkdir -p $LOG
MAXJOBS=2        # pathD's train pickle is 7.2 GB; two runs sit near 25 G of 125 G

BASE="data.data_dir=data/pathD/hermite_distinct/processed/geometry_split
      data.max_train_geometries=null data.max_train_trajectories=null
      data.max_val_trajectories=null wandb.enabled=false
      data.context.per_sample=false"

run () {   # $1=exp  $2=model-overrides  $3=arm  $4=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE $2 \
      data.train_geometry_ids=scratch/depth/train_$3.json \
      seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

echo "=== DEPTH-STRATIFIED B2 — START $(date '+%F %T') ==="
for S in 0 1 2; do
  for ARM in shallow deep; do
    run DS_${ARM}_pfn  "model=spatial_pfn model.readout=joint"   $ARM $S
    run DS_${ARM}_rcnp "model=spatial_rcnp"                      $ARM $S
    run DS_${ARM}_cnp  "model=spatial_cnp"                       $ARM $S
    run DS_${ARM}_cnpF "model=spatial_cnp model.attn_ffn=true"   $ARM $S
  done
done
wait
echo "=== ALL 24 RUNS DONE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, os, statistics as st
def best(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        lg=f"scratch/depth/{tag}_s{d.split('/seed_')[1]}.log"
        if not (os.path.exists(lg) and "done. best val mae" in open(lg).read()): continue
        r=[float(x["val_mae"]) for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
        if r: v.append(min(r))
    return v
print("\n  TRAINING-SET DEPTH BAND vs VALIDATION MAE (val MAE; suite pass still to run)\n")
print(f"  {'model':<8}{'full (PD_A)':>16}{'shallow-only':>16}{'deep-only':>14}")
for m,ref in (("pfn","PD_A_pfn"),("rcnp","PD_A_rcnp"),("cnp","PD_A_cnp"),("cnpF","PD_A_cnpF")):
    cells=[]
    for tag in (ref, f"DS_shallow_{m}", f"DS_deep_{m}"):
        v=best(tag)
        cells.append(f"{st.mean(v):.2f} ± {st.pstdev(v):.2f}" if v else "-")
    print(f"  {m:<8}{cells[0]:>16}{cells[1]:>16}{cells[2]:>14}")
print("\n  The decisive numbers are per-region, which needs the eval suite.")
EOF
