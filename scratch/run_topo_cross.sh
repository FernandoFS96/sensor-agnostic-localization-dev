#!/usr/bin/env bash
# Does restoring the CNP's RETRIEVAL bias fix the PFN on the topology task?
# Baseline this round: cnp 9 | pfn(joint) 71 | pfn(joint, lr3e-4) 136
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
C="dataset=topology_hermite data=topology data.topology=random data.trans_aug_m=0 \
   model=pfn training.epochs=1000 training.batch_size=128 training.num_workers=0 \
   training.lr=0.005 model.num_hidden=128 model.dropout=0.2 seed=1 wandb.enabled=false"
run(){ local n=$1; shift; echo "[x] $(date +%H:%M:%S) $n"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $C "$@" exp_name="$n" \
    > "scratch/${n}.log" 2>&1 & }
# the hypothesis: acoustics-only queries/keys, labels only in the values
run TOPO_pfn_cross      model.readout=cross
sleep 10
# shallower context encoder: is 6 layers too deep for 560 samples?
run TOPO_pfn_cross_L2   model.readout=cross model.n_layers=2
sleep 10
# control: keep the joint readout but shallow, to separate depth from readout
run TOPO_pfn_joint_L2   model.readout=joint model.n_layers=2
wait
echo "[x] $(date +%H:%M:%S) COMPLETE"
echo "===== TOPOLOGY hermite/random  (cnp 9 | pfn-joint 71) ====="
for n in TOPO_pfn_cross TOPO_pfn_cross_L2 TOPO_pfn_joint_L2; do
  d=$(ls -d output/hydra/training/*/$n/seed_1 2>/dev/null | tail -1)
  [ -f "$d/train_log.csv" ] && awk -F, -v N="$n" 'NR>1&&$9!=""{if(m==""||$9+0<m+0){m=$9;e=$1;tr=$5}}
    END{printf "  %-20s best val=%7.2f @ep%-5s (train %.1f)\n",N,m,e,tr}' "$d/train_log.csv"
done
