#!/usr/bin/env bash
# After both PD_B_ctxreg seeds finish: context controls (same transits and donors as
# b2.npz) and the calibration protocol with 0/3/5/7 sensors lost.
# Usage: eval_ctxreg_b2.sh <exp_name> <tag> [conditions]
cd /home/fer/tesis/sensor-agnostic-underwater-localization
name=$1; tag=$2
conds=${3:-calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,empty,perm_y,foreign,foreign_y,foreign_x,noise_x,shift_y10,shift_y30,self_prefix,self_spread}
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=${DD:-data/pathD/hermite_distinct/processed/geometry_split}
L=scratch/context_diag
for s in 0 1; do until grep -q "=== DONE" $L/train_${name}_s$s.log 2>/dev/null; do sleep 60; done; done
R0=$(ls -d output/hydra/training/*/$name/seed_0 | tail -1); R1=$(ls -d output/hydra/training/*/$name/seed_1 | tail -1)
echo "eval start $(date '+%F %T')"
$PY $L/ctx_controls.py --data-dir $DD --models s0=$R0/best.pt s1=$R1/best.pt --tag $tag \
    --conds $conds \
    > $L/$tag.log 2>&1
for k in 0 3 5 7; do
  TQDM_DISABLE=1 $PY scripts/eval/eval_calibration.py --data-dir $DD \
      --models ctxreg=$R0/best.pt ctxreg=$R1/best.pt --drop $k \
      --out-dir output/eval_paper/context_diag/${tag}_drop$k > $L/${tag}_drop$k.log 2>&1
done
echo "=== EVAL DONE $(date '+%F %T') ==="
