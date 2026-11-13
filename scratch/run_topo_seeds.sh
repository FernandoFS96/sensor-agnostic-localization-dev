#!/usr/bin/env bash
# Seed replication on the TOPOLOGY task (hermite/random).
# Seed 1 is already done (cnp 9, pfn_cross 8, pfn_joint_L2 9); add seeds 2 and 3.
# Run one SEED per batch (3 concurrent) -- 9 at once would exceed the 16 GB GPU.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
BASE="dataset=topology_hermite data=topology data.topology=random data.trans_aug_m=0 \
      training.epochs=1000 training.batch_size=128 training.num_workers=0 \
      training.lr=0.005 model.num_hidden=128 model.dropout=0.2 wandb.enabled=false"
batch(){
  local s=$1
  echo "[seeds] $(date +%H:%M:%S) seed $s"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE seed=$s model=cnp \
      exp_name=TOPO_cnp_rerun          > scratch/seed${s}_cnp.log 2>&1 &
  sleep 10
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE seed=$s model=pfn \
      model.readout=cross exp_name=TOPO_pfn_cross > scratch/seed${s}_cross.log 2>&1 &
  sleep 10
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE seed=$s model=pfn \
      model.readout=joint model.n_layers=2 exp_name=TOPO_pfn_joint_L2 \
      > scratch/seed${s}_jointL2.log 2>&1 &
  wait
  echo "[seeds] $(date +%H:%M:%S) seed $s done"
}
batch 2
batch 3
echo "[seeds] $(date +%H:%M:%S) ALL COMPLETE"
$PY - <<'PYEOF'
import glob, csv, statistics as st
print("\n===== TOPOLOGY hermite/random -- best val MAE over seeds =====")
print(f"  {'model':<22}{'seeds':<20}{'mean':>7}{'sd':>7}")
for n in ("TOPO_cnp_rerun", "TOPO_pfn_cross", "TOPO_pfn_joint_L2"):
    vals = []
    for d in sorted(glob.glob(f"output/hydra/training/*/{n}/seed_*")):
        try:
            v = [float(r["val_mae"]) for r in csv.DictReader(open(f"{d}/train_log.csv"))
                 if r.get("val_mae")]
            if v: vals.append(min(v))
        except Exception: pass
    if vals:
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {n:<22}{str([round(v,1) for v in vals]):<20}{st.mean(vals):7.2f}{sd:7.2f}")
PYEOF
