#!/usr/bin/env bash
# Queue after the PD_B_ctxreg seeds and their evaluation (eval_PD_B_ctxreg.log):
#   1. attention mode (spatial.ctx_mode=attn), two seeds, then its evaluation;
#   2. control with the context branch removed (spatial.ctx_gate=false), i.e. the
#      plain regressor inside the NP trainer, two seeds, then its evaluation.
# Evaluations run with the GPU otherwise free (two trainings fill it).
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
until grep -q "=== EVAL DONE" $L/eval_PD_B_ctxreg.log 2>/dev/null; do sleep 60; done
for s in 0 1; do
  $L/run_ctxreg_b2.sh $s PD_B_ctxreg_attn model.spatial.ctx_mode=attn > $L/train_PD_B_ctxreg_attn_s$s.log 2>&1 &
done
wait
$L/eval_ctxreg_b2.sh PD_B_ctxreg_attn b2_ctxreg_attn > $L/eval_PD_B_ctxreg_attn.log 2>&1
for s in 0 1; do
  $L/run_ctxreg_b2.sh $s PD_B_ctxreg_nogate model.spatial.ctx_gate=false > $L/train_PD_B_ctxreg_nogate_s$s.log 2>&1 &
done
wait
$L/eval_ctxreg_b2.sh PD_B_ctxreg_nogate b2_ctxreg_nogate calib_n10,empty > $L/eval_PD_B_ctxreg_nogate.log 2>&1
echo "=== CHAIN DONE $(date '+%F %T') ==="
