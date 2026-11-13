#!/usr/bin/env bash
# Prior-fitting on the analytic linear-array prior: grid head vs Gaussian head,
# same backbone, 3 seeds. Metric is KL against the EXACT posterior.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
for s in 0 1 2; do
  for h in grid gauss; do
    $PY pfn_lab/train.py --head $h --seed $s --steps 60000 \
        --out scratch/lab_${h}_s${s} > scratch/lab_${h}_s${s}.log 2>&1 &
  done
done
wait
echo "ALL COMPLETE"
$PY - <<'PYEOF'
import csv, statistics as st, glob, os
print(f"\n{'head':<8}{'seeds':<26}{'KL mean':>9}{'MAE mean':>10}{'axis-mass':>11}")
for h in ("grid","gauss"):
    kl,mae,ax = [],[],[]
    for d in sorted(glob.glob(f"scratch/lab_{h}_s*")):
        f=os.path.join(d,"log.csv")
        if not os.path.exists(f): continue
        r=list(csv.DictReader(open(f)))
        if r: kl.append(float(r[-1]["kl_exact"])); mae.append(float(r[-1]["mae"])); ax.append(float(r[-1]["axis_mass_model"]))
    if kl:
        print(f"{h:<8}{str([round(v,3) for v in kl]):<26}{st.mean(kl):9.3f}{st.mean(mae):10.1f}{st.mean(ax):11.3f}")
r=list(csv.DictReader(open("scratch/lab_grid_s0/log.csv")))
print(f"\n  exact posterior axis-mass reference: {float(r[-1]['axis_mass_exact']):.3f}")
PYEOF
