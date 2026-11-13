#!/usr/bin/env bash
# RANDOM topology at the agreed operating point: aperture 450 m, 250 Hz
# (sigma_tau = 4 ms) -> exact floor ~14.5 m. Three heads, 2 seeds. Small scale.
set -u
PY=/home/fer/miniconda3/envs/torch/bin/python
cd /home/fer/tesis/sensor-agnostic-underwater-localization
for s in 0 1; do for h in gauss full grid; do
  $PY pfn_lab/train.py --head $h --seed $s --steps 30000 --eval-tasks 64 \
      --layout random --sensors 10 --aperture 450 --sigma-tau 4e-3 --sigma-amp 0.40 \
      --out scratch/rnd_${h}_s${s} > scratch/rnd_${h}_s${s}.log 2>&1 &
done; done
wait
echo "ALL COMPLETE"
$PY - <<'PYEOF'
import csv, glob, os, statistics as st
print(f"\n  RANDOM topology, aperture 450 m, 250 Hz  (exact floor ~14.5 m)")
print(f"  {'head':<8}{'KL per seed':<20}{'KL mean':>9}")
for h in ("gauss","full","grid"):
    v=[]
    for d in sorted(glob.glob(f"scratch/rnd_{h}_s*")):
        f=os.path.join(d,"log.csv")
        if os.path.exists(f):
            r=list(csv.DictReader(open(f)))
            if r: v.append(float(r[-1]["kl_exact"]))
    if v: print(f"  {h:<8}{str([round(x,3) for x in v]):<20}{st.mean(v):>9.3f}")
print("\n  ceilings measured analytically: diagonal 0.74 | full-cov 0.15 | grid ~0")
PYEOF
