#!/usr/bin/env bash
# Full evaluation suite over every model trained on the CLEAN set.
#
# All 10 evaluators (tiers 1-4) under ONE common protocol, 3 evaluation seeds for
# error bars. The suite samples context as a deployment-CAUSAL first-prefix and
# has no calibration mode, so the CL_B_* models -- trained with cross-trajectory
# CALIBRATION context -- are scored OUT of their training regime here. They are
# included anyway because a single protocol is what makes the table comparable;
# their training-regime numbers stay in the Regime B column of the training table.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pfn10kd/hermite_distinct/processed/geometry_split
OUT=output/eval_suite_clean; mkdir -p "$OUT"
P(){ ls output/hydra/training/*/$1/seed_0/best.pt 2>/dev/null | tail -1; }

echo "===== REGIME A models, tiers 1-4 $(date '+%F %T') ====="
$PY scripts/eval/eval_suite.py --tiers 1,2,3,4 --eval-seeds 0,1,2 --cap-traj 6 \
  --data-dir $DD --out-dir "$OUT/regimeA" --models \
  cnp=$(P CL_A_cnp) cnp_ffn=$(P CL_A_cnpF) pfn=$(P CL_A_pfn) \
  ranp=$(P CL_A_ranp) rcnp=$(P CL_A_rcnp) > "$OUT/regimeA.log" 2>&1
echo "  rc=$? $(date '+%T')"

echo "===== REGIME B models, tiers 1-4 $(date '+%F %T') ====="
$PY scripts/eval/eval_suite.py --tiers 1,2,3,4 --eval-seeds 0,1,2 --cap-traj 6 \
  --data-dir $DD --out-dir "$OUT/regimeB" --models \
  cnp_cal=$(P CL_B_cnp) cnp_ffn_cal=$(P CL_B_cnpF) pfn_cal=$(P CL_B_pfn) \
  > "$OUT/regimeB.log" 2>&1
echo "  rc=$? $(date '+%T')"
echo "===== ALL EVAL DONE $(date '+%F %T') ====="
