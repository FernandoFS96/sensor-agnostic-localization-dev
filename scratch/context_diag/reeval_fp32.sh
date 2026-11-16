#!/usr/bin/env bash
# Section 13.9: every checkpoint of sections 13.7 and 13.8 again, in float32
# (eval_suite.load_model now switches bfloat16 off). Tags end in _fp32; the
# bfloat16 outputs keep their names.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
B1=data/pfn10kd/hermite_distinct/processed/geometry_split
S=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad
PFNC=calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,no_attn,perm_y,foreign,foreign_y,foreign_x,noise_x,shift_y10,shift_y30,self_prefix,self_spread
B1C=calib_n10,calib_n1,calib_n2,calib_n5,calib_n40,empty,perm_y,foreign,foreign_y,noise_x,shift_y10
$L/eval_ctxreg_b2.sh PD_B_ctxreg_attn_r80k   b2_ctxreg_attn_r80k_fp32                    > $L/eval_PD_B_ctxreg_attn_r80k_fp32.log 2>&1
$L/eval_ctxreg_b2.sh PD_B_ctxreg_nogate_r80k b2_ctxreg_nogate_r80k_fp32 calib_n10,empty  > $L/eval_PD_B_ctxreg_nogate_r80k_fp32.log 2>&1
$L/eval_ctxreg_b2.sh PD_B_pfnset_r80k        b2_pfnset_r80k_fp32 $PFNC                   > $L/eval_PD_B_pfnset_r80k_fp32.log 2>&1
echo "=== B2 DONE $(date '+%F %T') ==="
/home/fer/miniconda3/envs/torch/bin/python $L/reeval_regressors_fp32.py $S/acoustic_cache $S/acoustic_cache_b2 > $L/reeval_regressors_fp32.log 2>&1
echo "=== REGRESSORS DONE $(date '+%F %T') ==="
DD=$B1 $L/eval_ctxreg_b2.sh CL_B_ctxreg_attn_r80k   b1_ctxreg_attn_r80k_fp32 $B1C            > $L/eval_CL_B_ctxreg_attn_r80k_fp32.log 2>&1
DD=$B1 $L/eval_ctxreg_b2.sh CL_B_ctxreg_nogate_r80k b1_ctxreg_nogate_r80k_fp32 calib_n10,empty > $L/eval_CL_B_ctxreg_nogate_r80k_fp32.log 2>&1
echo "=== B1 DONE $(date '+%F %T') ==="
# The retraining test with the float32 head (one seed).
until grep -q "=== DONE" $L/train_PD_B_ctxreg_attn_r80k_h32_s0.log 2>/dev/null; do sleep 60; done
H=$(ls -d output/hydra/training/*/PD_B_ctxreg_attn_r80k_h32/seed_0 | tail -1)
/home/fer/miniconda3/envs/torch/bin/python $L/ctx_controls.py --data-dir data/pathD/hermite_distinct/processed/geometry_split \
    --models h32=$H/best.pt --tag b2_ctxreg_attn_r80k_h32 \
    --conds calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,empty,perm_y,foreign,shift_y10,self_prefix \
    > $L/b2_ctxreg_attn_r80k_h32.log 2>&1
for k in 0 3 5 7; do
  TQDM_DISABLE=1 /home/fer/miniconda3/envs/torch/bin/python scripts/eval/eval_calibration.py --data-dir data/pathD/hermite_distinct/processed/geometry_split \
      --models ctxreg=$H/best.pt --drop $k \
      --out-dir output/eval_paper/context_diag/b2_ctxreg_attn_r80k_h32_drop$k > $L/b2_ctxreg_attn_r80k_h32_drop$k.log 2>&1
done
echo "=== ALL DONE $(date '+%F %T') ==="
