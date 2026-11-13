#!/usr/bin/env bash
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
for s in 0 1 2; do for h in grid gauss; do
  $PY pfn_lab/train.py --head $h --seed $s --steps 60000 --eval-tasks 64 \
      --sensors 8 --sigma-tau 8e-3 --sigma-amp 0.40 \
      --out scratch/v2_${h}_s${s} > scratch/v2_${h}_s${s}.log 2>&1 &
done; done
wait
echo "ALL COMPLETE"
$PY - <<'PYEOF'
import csv, glob, os, statistics as st
print(f"\n  {'head':<8}{'KL per seed':<28}{'KL mean':>9}{'axis-mass':>11}{'MAE':>8}")
for h in ("grid","gauss"):
    kl,ax,mae=[],[],[]
    for d in sorted(glob.glob(f"scratch/v2_{h}_s*")):
        f=os.path.join(d,"log.csv")
        if not os.path.exists(f): continue
        r=list(csv.DictReader(open(f)))
        if r: kl.append(float(r[-1]["kl_exact"])); ax.append(float(r[-1]["axis_mass_model"])); mae.append(float(r[-1]["mae"]))
    if kl: print(f"  {h:<8}{str([round(v,3) for v in kl]):<28}{st.mean(kl):>9.3f}{st.mean(ax):>11.3f}{st.mean(mae):>8.1f}")
r=list(csv.DictReader(open("scratch/v2_grid_s0/log.csv")))
print(f"\n  exact-posterior axis-mass reference: {float(r[-1]['axis_mass_exact']):.3f}")
PYEOF
