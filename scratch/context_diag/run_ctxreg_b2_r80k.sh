#!/usr/bin/env bash
# The regressor's budget inside the NP trainer: 48 transits per batch (400 steps
# per epoch, 80k steps in 200 epochs), lr 3e-4 with a 10-epoch linear warm-up.
# No early stopping. Otherwise as run_ctxreg_b2.sh. Usage: run_ctxreg_b2_r80k.sh <seed> <exp_name> [overrides]
seed=$1; name=$2; shift 2
exec /home/fer/tesis/sensor-agnostic-underwater-localization/scratch/context_diag/run_ctxreg_b2.sh $seed $name \
    training.batch_size=48 training.lr=3e-4 training.warmup_epochs=10 \
    training.early_stop_patience=0 "$@"
