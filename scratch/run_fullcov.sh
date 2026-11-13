#!/usr/bin/env bash
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
BASE="dataset=topology_hermite data=topology data.topology=random data.trans_aug_m=0 \
      training.epochs=1000 training.batch_size=128 training.num_workers=0 \
      training.lr=0.005 model=cnp model.num_hidden=128 model.dropout=0.2 wandb.enabled=false"
for s in 1 2 3; do
  for FC in false true; do
    TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE seed=$s \
        model.full_cov=$FC exp_name=FC_cnp_${FC} > scratch/FC_cnp_${FC}_s${s}.log 2>&1 &
  done
  wait
done
echo "ALL COMPLETE"
$PY - <<'PYEOF'
import csv, glob, numpy as np
print(f"\n  TOPOLOGY hermite/random -- CNP, diagonal vs correlated (x,y) head")
print(f"  {'head':<16}{'val MAE per seed':<26}{'MAE mean':>10}{'val NLL':>10}")
for fc,lab in (("false","diagonal"),("true","full-cov")):
    mae,nll=[],[]
    for d in sorted(glob.glob(f"output/hydra/training/*/FC_cnp_{fc}/seed_*")):
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r:
                b=min(r,key=lambda x:float(x["val_mae"]))
                mae.append(float(b["val_mae"])); nll.append(float(b["val_nll"]))
        except Exception: pass
    if mae:
        print(f"  {lab:<16}{str([round(v,1) for v in mae]):<26}{np.mean(mae):>10.2f}{np.mean(nll):>10.3f}")
PYEOF
