#!/usr/bin/env bash
# KL plateaued at ~1.25 with no fixed dataset to overfit -> underfitting.
# Sweep capacity at a fixed budget; bigger eval set so the metric is trustworthy.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
run(){ $PY pfn_lab/train.py --head grid --seed 0 --steps 60000 --eval-tasks 64 \
        --layers $1 --hidden $2 --cross $3 --out scratch/cap_L$1H$2X$3 \
        > scratch/cap_L$1H$2X$3.log 2>&1 & }
run 4 128 2      # current baseline, re-measured on the bigger eval set
run 8 128 2      # deeper
run 4 256 2      # wider
run 8 256 4      # both + more cross layers
wait
echo "ALL COMPLETE"
$PY - <<'PYEOF'
import csv, glob, os
print(f"\n  {'config':<16}{'params':>9}{'final KL':>10}{'best KL':>9}{'axis':>8}")
for d in sorted(glob.glob("scratch/cap_*")):
    if not os.path.isdir(d): continue
    f=os.path.join(d,"log.csv")
    if not os.path.exists(f): continue
    r=list(csv.DictReader(open(f)))
    if not r: continue
    kl=[float(x["kl_exact"]) for x in r]
    print(f"  {os.path.basename(d):<16}{'':>9}{kl[-1]:>10.3f}{min(kl):>9.3f}{float(r[-1]['axis_mass_model']):>8.3f}")
print("\n  reference: gaussian head 3.20 (structural floor) | exact axis-mass 0.073")
PYEOF
