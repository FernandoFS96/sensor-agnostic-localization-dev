#!/usr/bin/env bash
# STAGE 1: does adding the missing feed-forward sublayer fix the CNP?
# STAGE 2: the full family table on the 8000-layout set.
#
# BACKGROUND. On this dataset the CNP reaches val 12.47 / train 21.88 where the
# PFN backbone reaches 8.04 / 10.13, and a capacity-matched ablation (Path B)
# put ~74% of that gap on architecture rather than size. The concrete difference
# is that anp.Attention has NO feed-forward sublayer and is 2 layers deep against
# the PFN's 6. attn_ffn=true adds x + FFN(LayerNorm(x)) to each attention block
# and changes nothing else.
#
# STAGE 1 is the decisive cheap test: if attn_ffn moves the CNP from 12.47
# toward ~8, one block fix explains and repairs the whole anp family, which is a
# far more useful result than a table of four models sharing one defect.
#
# STAGE 2 CAVEAT, stated up front: anp / ranp / rcnp cannot use per_sample=true
# (whitelisted to cnp/pfn -- only the deterministic split model handles the
# padding masks), so they run at per_sample=false. That is a DIFFERENT regime
# from the PT_ results, so the table includes cnp at per_sample=false as its own
# anchor and must be read within itself, not against 8.04 / 12.47.
# online_ranp is excluded entirely: the spatial encoder is not wired for it and
# this task requires one (trans_aug translates sensors and labels together).
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pfn10k/hermite_shared/processed/geometry_split
LOG=scratch/ffn; mkdir -p $LOG
MAXJOBS=3

BASE="data.data_dir=$DD data.max_train_geometries=null
      data.max_train_trajectories=null data.max_val_trajectories=null
      wandb.enabled=false"

run () {   # $1=exp $2=model $3=extra $4=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 $3 seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

# ---------------- STAGE 1: the retrofit ---------------------------------------
echo "=== STAGE 1 -- FFN RETROFIT $(date '+%F %T') ==="
for S in 0 1; do run FFN_cnp_on spatial_cnp "model.attn_ffn=true" $S; done
wait
echo "=== STAGE 1 COMPLETE $(date '+%F %T') ==="
$PY - <<'EOF'
import csv, glob, numpy as np
def best(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append((min(float(x["val_mae"]) for x in r), min(float(x["train_mae"]) for x in r)))
        except Exception: pass
    return v
print("\n  STAGE 1 -- does the missing FFN explain the gap?\n")
print(f"  {'arm':<36}{'val MAE':>9}{'train MAE':>11}")
for tag,lbl in (("PT_cnp","spatial_CNP (original block)"),
                ("FFN_cnp_on","spatial_CNP + FFN retrofit"),
                ("PT_joint","spatial_PFN joint (target)")):
    v=best(tag)
    if v: print(f"  {lbl:<36}{np.mean([x[0] for x in v]):>9.2f}{np.mean([x[1] for x in v]):>11.2f}")
EOF

# ---------------- STAGE 2: the family table -----------------------------------
# Every arm at per_sample=false so the table is internally consistent, cnp
# included as its own anchor.
echo ""
echo "=== STAGE 2 -- FAMILY TABLE $(date '+%F %T') ==="
PS="data.context.per_sample=false"
for S in 0; do
  run FAM_cnp   spatial_cnp  "$PS"                      $S
  run FAM_anp   spatial_anp  "$PS"                      $S
  run FAM_rcnp  spatial_rcnp "$PS"                      $S
  run FAM_ranp  spatial_ranp "$PS"                      $S
  run FAM_pfn   spatial_pfn  "$PS model.readout=joint"  $S
  run FAM_cnpF  spatial_cnp  "$PS model.attn_ffn=true"  $S
done
wait
echo "=== STAGE 2 COMPLETE $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, numpy as np
def best(tag):
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append((min(float(x["val_mae"]) for x in r), min(float(x["train_mae"]) for x in r)))
        except Exception: pass
    return v
print("\n  FAMILY TABLE -- 8000 layouts, per_sample=false (internally comparable)\n")
print(f"  {'model':<34}{'val MAE':>9}{'train MAE':>11}")
for tag,lbl in (("FAM_cnp","spatial_CNP"),("FAM_anp","spatial_ANP (latent)"),
                ("FAM_rcnp","spatial_RCNP (recurrent)"),("FAM_ranp","spatial_RANP (recurrent latent)"),
                ("FAM_cnpF","spatial_CNP + FFN retrofit"),("FAM_pfn","spatial_PFN (joint)")):
    v=best(tag)
    print(f"  {lbl:<34}" + (f"{np.mean([x[0] for x in v]):>9.2f}{np.mean([x[1] for x in v]):>11.2f}"
                            if v else f"{'-':>9}{'-':>11}"))
print("\n  (online_ranp excluded: spatial encoder not wired for it, required by this task)")
EOF
