#!/usr/bin/env bash
# Queue behind the two r80k runs (attention s0, gate-closed s0):
#   when the gate-closed run ends, start attention seed 1 in the free slot;
#   when attention s0 ends, score attention s0 and gate-closed s0 (first look);
#   when attention s1 ends, the full evaluation of both attention seeds.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/pathD/hermite_distinct/processed/geometry_split
until grep -q "=== DONE" $L/train_PD_B_ctxreg_nogate_r80k_s0.log 2>/dev/null; do sleep 60; done
$L/run_ctxreg_b2_r80k.sh 1 PD_B_ctxreg_attn_r80k model.spatial.ctx_mode=attn > $L/train_PD_B_ctxreg_attn_r80k_s1.log 2>&1 &
until grep -q "=== DONE" $L/train_PD_B_ctxreg_attn_r80k_s0.log 2>/dev/null; do sleep 60; done
A=$(ls -d output/hydra/training/*/PD_B_ctxreg_attn_r80k/seed_0 | tail -1); G=$(ls -d output/hydra/training/*/PD_B_ctxreg_nogate_r80k/seed_0 | tail -1)
$PY $L/ctx_controls.py --data-dir $DD --models attn=$A/best.pt nogate=$G/best.pt --tag b2_r80k_s0 \
    --conds calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,empty > $L/b2_r80k_s0.log 2>&1
for k in 0 3 5 7; do
  TQDM_DISABLE=1 $PY scripts/eval/eval_calibration.py --data-dir $DD --models nogate=$G/best.pt --drop $k \
      --out-dir output/eval_paper/context_diag/b2_ctxreg_nogate_r80k_drop$k > $L/b2_ctxreg_nogate_r80k_drop$k.log 2>&1
done
echo "=== FIRST LOOK DONE $(date '+%F %T') ==="
wait
$L/eval_ctxreg_b2.sh PD_B_ctxreg_attn_r80k b2_ctxreg_attn_r80k > $L/eval_PD_B_ctxreg_attn_r80k.log 2>&1
echo "=== CHAIN DONE $(date '+%F %T') ==="
