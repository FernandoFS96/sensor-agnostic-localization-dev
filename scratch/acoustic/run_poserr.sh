#!/usr/bin/env bash
# Does the PFN learn to USE its context once sensor-position error is in its training
# prior? B2, both models trained with sd ~ U(0, 6) m per sample (data.sensor_pos_err_m /
# --pos-err), otherwise the fixdrop / nll_drop recipes. In parallel:
#   PD_B_pfnset_poserr s0 (PFN, calibration regime, 200 epochs)
#   regressor tone nll_drop_poserr (80k steps; its robustness csv covers pos err 2/5 m)
# Then eval_calibration of old fixdrop vs new (best + last) at pos err 0/2/5/8 m,
# a context-size sweep at 5 m (1/2/10/40), and the wrong-context control (donor from
# another deployment) at 0 and 5 m.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
SCR=/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad
DD=data/pathD/hermite_distinct/processed/geometry_split
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
echo "start $(date '+%F %T')"
$PY scripts/train/train_np_geometry.py data.data_dir=$DD \
    data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
    model=spatial_pfn_set model.readout=joint data.rot_aug=true \
    data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
    data.sensor_pos_err_m=6 \
    training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=PD_B_pfnset_poserr seed=0 \
    > scratch/acoustic/PD_B_pfnset_poserr_s0.log 2>&1 &
PFN=$!
$PY scripts/eval/acoustic_regressor.py train --cache $SCR/acoustic_cache_b2 --data-dir $DD --feat tone \
    --steps 80000 --loss nll --sensor-drop 0.3 --pos-err 6 --tag nll_drop_poserr \
    --out-dir output/eval_paper/baselines/b2 > scratch/acoustic/train_b2_tone_nll_drop_poserr.log 2>&1
echo "regressor done $(date '+%F %T')"
wait $PFN
echo "pfn done $(date '+%F %T')"
O=$(ls -d output/hydra/training/*/PD_B_pfnset_fixdrop/seed_0 | tail -1)
N=$(ls -d output/hydra/training/*/PD_B_pfnset_poserr/seed_0 | tail -1)
M="fixdrop=$O/best.pt poserr=$N/best.pt poserr_last=$N/last.pt"
EV() { $PY scripts/eval/eval_calibration.py --data-dir $DD --models $M "$@"; }
for e in 0 2 5 8; do
  EV --pos-err $e --out-dir output/eval_paper/b2_poserr/pos$e > scratch/acoustic/eval_poserr_pos$e.log 2>&1
done
for k in 1 2 40; do
  EV --pos-err 5 --n-ctx $k --out-dir output/eval_paper/b2_poserr/pos5_ctx$k > scratch/acoustic/eval_poserr_pos5_ctx$k.log 2>&1
done
for e in 0 5; do
  EV --pos-err $e --foreign-donor --out-dir output/eval_paper/b2_poserr/pos${e}_foreign > scratch/acoustic/eval_poserr_pos${e}_foreign.log 2>&1
done
echo "=== DONE $(date '+%F %T') ==="
