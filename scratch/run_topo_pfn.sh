#!/usr/bin/env bash
# PFN vs CNP on the TOPOLOGY task (hermite, random topology), matched protocol.
# Existing baselines on this data: cnp 9, rcnp 14, ranp 15, anp 39.
# Protocol copied from the 2026-07-21 CNP run: 1000 ep, batch 128, lr 5e-3,
# ctx 2/40/eval 10, mode=first, per_sample, dropout 0.2, no augmentation.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
C="dataset=topology_hermite data=topology data.topology=random data.trans_aug_m=0 \
   training.epochs=1000 training.batch_size=128 training.num_workers=0 \
   training.lr=0.005 model.num_hidden=128 model.dropout=0.2 seed=1 wandb.enabled=false"
run(){ local n=$1; shift; echo "[topo] $(date +%H:%M:%S) $n"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C "$@" exp_name="$n" \
    > "scratch/${n}.log" 2>&1 & }
# matched-protocol PFN (same lr as the CNP baseline)
run TOPO_pfn_lr5e3   model=pfn
sleep 10
# lr=5e-3 is very high for a pre-LN transformer; give it a standard one too
run TOPO_pfn_lr3e4   model=pfn training.lr=3e-4
sleep 10
# CNP re-run under the CURRENT code, so the comparison is same-session
run TOPO_cnp_rerun   model=cnp
wait
echo "[topo] $(date +%H:%M:%S) COMPLETE"
echo "===== TOPOLOGY task, hermite/random (existing: cnp 9, rcnp 14, ranp 15, anp 39) ====="
for n in TOPO_cnp_rerun TOPO_pfn_lr5e3 TOPO_pfn_lr3e4; do
  d=$(ls -d output/hydra/training/*/$n/seed_1 2>/dev/null | tail -1)
  [ -f "$d/train_log.csv" ] && awk -F, -v N="$n" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1}; lt=$5; l=$1}
    END{printf "  %-18s best val=%7.2f @ep%-5s (train %.1f, stopped ep%s)\n",N,m,e,lt,l}' "$d/train_log.csv"
done
