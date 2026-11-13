#!/usr/bin/env bash
# =============================================================================
# Paper sweep: missing models + spatial ablation ladder + spatial_ranp fusion.
# Runs SEQUENTIALLY (one job at a time -> one GPU, no contention). Each job
# early-stops, logs to output/sweep_logs/<name>.log, and a failure does NOT stop
# the batch. Survives SSH disconnect when launched detached (see bottom).
#
# LAUNCH (pick one), from the repo root:
#   tmux new -s sweep 'bash scripts/train/run_paper_sweep.sh'   # reattach: tmux attach -t sweep
#   nohup bash scripts/train/run_paper_sweep.sh > output/sweep_logs/sweep.out 2>&1 &  disown
# Watch:  tail -f output/sweep_logs/sweep.out   (or a specific <name>.log)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."                       # repo root
PY=/home/fer/miniconda3/envs/torch/bin/python
LOGDIR=output/sweep_logs; mkdir -p "$LOGDIR"

# Shared, matched settings so every model is comparable (128-d, standard
# per-batch context, offline wandb so no connectivity is needed). Spatial models
# add trans_aug_m=150 and inherit n_attn_layers=3 from config/model/_spatial.yaml;
# flat/recurrent models use aug=0 and require per_sample=false (guarded).
# Seeds to train (override: SEEDS="1 2" bash scripts/train/run_paper_sweep.sh).
# seed=0 is already done for most jobs; the per-seed idempotency check skips
# completed (name,seed) pairs, so a full 0 1 2 relaunch only trains what's missing.
SEEDS="${SEEDS:-0 1 2}"
COMMON="data=layout_ood training.num_workers=4 wandb.offline=true \
        model.num_hidden=128 data.context.per_sample=false"

# name :: per-job overrides (dataset defaults to hermite unless overridden)
JOBS=(
  # ---- hermite layout_ood: full 2x3 matrix + fusion -----------------------
  "flat_cnp::model=cnp         dataset=random_hermite_shared data.trans_aug_m=0"
  "flat_anp::model=anp         dataset=random_hermite_shared data.trans_aug_m=0"
  "spatial_cnp_full::model=spatial_cnp dataset=random_hermite_shared data.trans_aug_m=150"
  "spatial_anp::model=spatial_anp dataset=random_hermite_shared data.trans_aug_m=150"
  "rcnp::model=rcnp            dataset=random_hermite_shared data.trans_aug_m=0"
  "ranp::model=ranp            dataset=random_hermite_shared data.trans_aug_m=0"
  "spatial_ranp::model=spatial_ranp dataset=random_hermite_shared data.trans_aug_m=150"
  # ---- spatial ablation ladder (all spatial_cnp, aug150) ------------------
  "abl_notokenize::model=spatial_cnp dataset=random_hermite_shared data.trans_aug_m=150 model.spatial.tokenize=false"
  "abl_nopos::model=spatial_cnp      dataset=random_hermite_shared data.trans_aug_m=150 model.spatial.use_position=false"
  "abl_noattn::model=spatial_cnp     dataset=random_hermite_shared data.trans_aug_m=150 model.spatial.use_attention=false"
  "abl_meanpool::model=spatial_cnp   dataset=random_hermite_shared data.trans_aug_m=150 model.spatial.pooling=mean"
  # ---- cross-dataset baselines (spiral layout_ood) ------------------------
  "flat_cnp_spiral::model=cnp  dataset=random_spiral_shared data.trans_aug_m=0"
  "flat_anp_spiral::model=anp  dataset=random_spiral_shared data.trans_aug_m=0"
  "spatial_cnp_spiral::model=spatial_cnp dataset=random_spiral_shared data.trans_aug_m=150"
)

N=$(( ${#JOBS[@]} * $(echo $SEEDS | wc -w) )); i=0
echo "[sweep] $N (job x seed) runs over seeds [$SEEDS] | start $(date)"
for seed in $SEEDS; do
for job in "${JOBS[@]}"; do
  i=$((i+1))
  name="${job%%::*}"; ovr="${job#*::}"
  log="$LOGDIR/${name}_s${seed}.log"
  # Resumable: last.pt is saved only on completion; skip finished (name,seed) pairs.
  if ls output/hydra/training/*/paper_${name}/seed_${seed}/last.pt >/dev/null 2>&1; then
    echo "[sweep] ($i/$N) SKIP  ${name} seed=$seed (already completed)"
    continue
  fi
  echo "[sweep] ($i/$N) START ${name} seed=$seed  ->  $log  $(date)"
  if $PY scripts/train/train_np_geometry.py $COMMON seed=$seed $ovr exp_name="paper_${name}" \
        > "$log" 2>&1; then
    best=$(grep -oE 'best val mae=[0-9.]+' "$log" | tail -1)
    echo "[sweep] ($i/$N) DONE  ${name} seed=$seed  ${best:-(no best line)}  $(date)"
  else
    echo "[sweep] ($i/$N) FAIL  ${name} seed=$seed  (tail: $(tail -1 "$log"))  $(date)"
  fi
done
done
echo "[sweep] ALL DONE $(date)"
