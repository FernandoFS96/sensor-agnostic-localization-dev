#!/usr/bin/env bash
# Chirp B2 with the 80k recipe (approved 16 November 2026): the lean attention model C and its
# gate-closed ablation, two seeds each, two runs at a time. Data: pathD_chirp (same layout,
# environments, trajectories and split as B2; 4 delay features per sensor).
#   PD_B_ctxreg_attn_lean_chirp_r80k   C (6.10 M with tone records; smaller input map here)
#   PD_B_ctxreg_nogate_chirp_r80k      gate closed (plain regressor in the NP trainer)
# Evaluation tags b2c_ctxreg_lean_r80k and b2c_ctxreg_nogate_r80k (controls + 0/3/5/7 sensors lost).
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
export TRAIN_DD=data/pathD_chirp/hermite_distinct/processed_delay/geometry_split
export DD=$TRAIN_DD
CONDS=calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,empty,perm_y,foreign,shift_y10,self_prefix
echo "queue start $(date '+%F %T')"
for s in 0 1; do
  $L/run_ctxreg_b2_r80k.sh $s PD_B_ctxreg_attn_lean_chirp_r80k model.spatial.ctx_mode=attn model.spatial.ctx_share=true \
      model.spatial.ctx_pool_layers=0 'model.spatial.ctx_attn_layers=[1,2,5,6]' \
      > $L/train_PD_B_ctxreg_attn_lean_chirp_r80k_s$s.log 2>&1 &
done
wait
echo "lean pair trained $(date '+%F %T')"
for s in 0 1; do
  $L/run_ctxreg_b2_r80k.sh $s PD_B_ctxreg_nogate_chirp_r80k model.spatial.ctx_gate=false \
      > $L/train_PD_B_ctxreg_nogate_chirp_r80k_s$s.log 2>&1 &
done
$L/eval_ctxreg_b2.sh PD_B_ctxreg_attn_lean_chirp_r80k b2c_ctxreg_lean_r80k $CONDS > $L/eval_PD_B_ctxreg_attn_lean_chirp_r80k.log 2>&1
echo "=== CHIRP LEAN EVAL DONE $(date '+%F %T') ==="
wait
echo "gate-closed pair trained $(date '+%F %T')"
$L/eval_ctxreg_b2.sh PD_B_ctxreg_nogate_chirp_r80k b2c_ctxreg_nogate_r80k calib_n10,empty > $L/eval_PD_B_ctxreg_nogate_chirp_r80k.log 2>&1
echo "=== CHIRP DONE $(date '+%F %T') ==="
