#!/usr/bin/env bash
# Second seed of the lean attention model C (section 13.10): shared sensor encoder, no context
# transformer, context attention before layers 1, 2, 5, 6 (6.10 M). B2, 80k recipe, seed 1.
# Then the context controls and the sensor-loss protocol for both seeds (tag b2_ctxreg_lean_r80k).
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/context_diag
$L/run_ctxreg_b2_r80k.sh 1 PD_B_ctxreg_attn_lean_r80k model.spatial.ctx_mode=attn model.spatial.ctx_share=true \
    model.spatial.ctx_pool_layers=0 'model.spatial.ctx_attn_layers=[1,2,5,6]' \
    > $L/train_PD_B_ctxreg_attn_lean_r80k_s1.log 2>&1
$L/eval_ctxreg_b2.sh PD_B_ctxreg_attn_lean_r80k b2_ctxreg_lean_r80k
echo "=== LEAN S1 DONE $(date '+%F %T') ==="
