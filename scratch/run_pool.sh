#!/usr/bin/env bash
# Q1 + Q3: at a FIXED random layout, how much does the size of the recycled task
# pool matter? pool=0 is true prior-fitting (fresh every batch).
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
for P in 10 100 1000 10000 0; do
  $PY pfn_lab/train.py --head full --seed 0 --steps 30000 --eval-tasks 64 \
      --layout random --sensors 10 --aperture 450 --sigma-tau 4e-3 --sigma-amp 0.40 \
      --pool $P --out scratch/pool_${P} > scratch/pool_${P}.log 2>&1 &
done
wait
echo "ALL COMPLETE"
$PY - <<'PYEOF'
import csv, os
print(f"\n  Fixed RANDOM layout, full-cov head, 30k steps")
print(f"  {'task pool':<18}{'KL(exact||model)':>18}{'MAE':>9}")
for P in (10,100,1000,10000,0):
    f=f"scratch/pool_{P}/log.csv"
    if not os.path.exists(f): continue
    r=list(csv.DictReader(open(f)))
    if r:
        lab = "fresh every batch" if P==0 else f"{P} tasks recycled"
        print(f"  {lab:<18}{float(r[-1]['kl_exact']):>18.3f}{float(r[-1]['mae']):>8.1f}m")
PYEOF
