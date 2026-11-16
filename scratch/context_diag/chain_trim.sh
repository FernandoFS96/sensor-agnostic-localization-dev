#!/usr/bin/env bash
# Lighter attention models (section 13.10), B2, 80k recipe, seed 0, side by side:
#   A  PD_B_ctxreg_attn_share_r80k  one sensor encoder for context and target points (8.21 M)
#   C  PD_B_ctxreg_attn_lean_r80k   A + no context transformer + attention before layers 1, 2, 5, 6 (6.10 M)
# then the context controls and the sensor-loss protocol for both.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pathD/hermite_distinct/processed/geometry_split
$L/run_ctxreg_b2_r80k.sh 0 PD_B_ctxreg_attn_share_r80k model.spatial.ctx_mode=attn model.spatial.ctx_share=true \
    > $L/train_PD_B_ctxreg_attn_share_r80k_s0.log 2>&1 &
$L/run_ctxreg_b2_r80k.sh 0 PD_B_ctxreg_attn_lean_r80k model.spatial.ctx_mode=attn model.spatial.ctx_share=true \
    model.spatial.ctx_pool_layers=0 'model.spatial.ctx_attn_layers=[1,2,5,6]' \
    > $L/train_PD_B_ctxreg_attn_lean_r80k_s0.log 2>&1 &
wait
A=$(ls -d output/hydra/training/*/PD_B_ctxreg_attn_share_r80k/seed_0 | tail -1)
C=$(ls -d output/hydra/training/*/PD_B_ctxreg_attn_lean_r80k/seed_0 | tail -1)
$PY $L/ctx_controls.py --data-dir $DD --models share=$A/best.pt lean=$C/best.pt --tag b2_ctxreg_trim \
    --conds calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,empty,perm_y,foreign,shift_y10,self_prefix \
    > $L/b2_ctxreg_trim.log 2>&1
for k in 0 3 5 7; do
  TQDM_DISABLE=1 $PY scripts/eval/eval_calibration.py --data-dir $DD --models share=$A/best.pt lean=$C/best.pt --drop $k \
      --out-dir output/eval_paper/context_diag/b2_ctxreg_trim_drop$k > $L/b2_ctxreg_trim_drop$k.log 2>&1
done
echo "=== TRIM DONE $(date '+%F %T') ==="
