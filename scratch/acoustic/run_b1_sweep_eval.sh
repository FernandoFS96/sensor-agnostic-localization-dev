#!/usr/bin/env bash
# B1 size sweep evaluation without width 384 (dropped 2026-11-06: no gain on B2).
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pfn10kd/hermite_distinct/processed/geometry_split
ck() { ls -d output/hydra/training/*/$1/seed_0 2>/dev/null | tail -1; }
M=""
for r in d128_l6 d192_l6 d256_l3 d256_l6; do d=$(ck CL_B_pfnset_$r); [ -f $d/best.pt ] && M="$M $r=$d/best.pt"; done
echo "models:$M"
for k in 0 3 5 7; do
  $PY scripts/eval/eval_calibration.py --data-dir $DD --models $M --drop $k \
      --out-dir output/eval_paper/b1_pfn_size_sweep_drop$k > scratch/acoustic/eval_b1_sweep_drop$k.log 2>&1
  echo "drop$k done $(date '+%F %T') (exit $?)"
done
echo "=== DONE $(date '+%F %T') ==="
