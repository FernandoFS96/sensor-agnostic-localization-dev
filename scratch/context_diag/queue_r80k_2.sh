#!/usr/bin/env bash
# Second batch with the regressor's budget (section 13.7), two runs at a time:
#   lane A: B2 gate-closed s1 -> B2 original PFN s1 -> B1 gate-closed s0 -> B1 gate-closed s1
#   lane B: B2 original PFN s0 -> B1 attention s0 -> B1 attention s1
# then evaluations as each pair completes (they share the GPU with training).
# B1 has 48 000 training transits: 48 per batch = 1000 steps per epoch, so 80
# epochs (80k steps) with a 4-epoch warm-up.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
B1=data/pfn10kd/hermite_distinct/processed/geometry_split
b2() { $L/run_ctxreg_b2_r80k.sh "$@"; }                       # <seed> <name> [overrides]
b1() { $L/run_ctxreg_b2_r80k.sh "$@" data.data_dir=$B1 training.epochs=80 training.warmup_epochs=4; }
pfn() { MODEL=spatial_pfn_set b2 "$@" model.readout=joint; }

(  b2  1 PD_B_ctxreg_nogate_r80k model.spatial.ctx_gate=false > $L/train_PD_B_ctxreg_nogate_r80k_s1.log 2>&1
   pfn 1 PD_B_pfnset_r80k                                     > $L/train_PD_B_pfnset_r80k_s1.log 2>&1
   b1  0 CL_B_ctxreg_nogate_r80k model.spatial.ctx_gate=false > $L/train_CL_B_ctxreg_nogate_r80k_s0.log 2>&1
   b1  1 CL_B_ctxreg_nogate_r80k model.spatial.ctx_gate=false > $L/train_CL_B_ctxreg_nogate_r80k_s1.log 2>&1 ) &
(  pfn 0 PD_B_pfnset_r80k                                     > $L/train_PD_B_pfnset_r80k_s0.log 2>&1
   b1  0 CL_B_ctxreg_attn_r80k model.spatial.ctx_mode=attn    > $L/train_CL_B_ctxreg_attn_r80k_s0.log 2>&1
   b1  1 CL_B_ctxreg_attn_r80k model.spatial.ctx_mode=attn    > $L/train_CL_B_ctxreg_attn_r80k_s1.log 2>&1 ) &

# Evaluations (eval_ctxreg_b2.sh waits for both seeds of a run name).
$L/eval_ctxreg_b2.sh PD_B_ctxreg_nogate_r80k b2_ctxreg_nogate_r80k calib_n10,empty > $L/eval_PD_B_ctxreg_nogate_r80k.log 2>&1
echo "=== B2 GATE-CLOSED DONE $(date '+%F %T') ==="
$L/eval_ctxreg_b2.sh PD_B_pfnset_r80k b2_pfnset_r80k \
    calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,no_attn,perm_y,foreign,foreign_y,foreign_x,noise_x,shift_y10,shift_y30,self_prefix,self_spread \
    > $L/eval_PD_B_pfnset_r80k.log 2>&1
echo "=== B2 PFN DONE $(date '+%F %T') ==="
B1C=calib_n10,calib_n1,calib_n2,calib_n5,calib_n40,empty,perm_y,foreign,foreign_y,noise_x,shift_y10
DD=$B1 $L/eval_ctxreg_b2.sh CL_B_ctxreg_attn_r80k b1_ctxreg_attn_r80k $B1C > $L/eval_CL_B_ctxreg_attn_r80k.log 2>&1
echo "=== B1 ATTENTION DONE $(date '+%F %T') ==="
DD=$B1 $L/eval_ctxreg_b2.sh CL_B_ctxreg_nogate_r80k b1_ctxreg_nogate_r80k calib_n10,empty > $L/eval_CL_B_ctxreg_nogate_r80k.log 2>&1
echo "=== B1 GATE-CLOSED DONE $(date '+%F %T') ==="
wait
echo "=== QUEUE DONE $(date '+%F %T') ==="
