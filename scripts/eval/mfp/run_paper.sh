#!/usr/bin/env bash
# Matched-field processing for the paper (Appendix F, Table tab:mfp), paper protocol:
# points 11-50 of every held-out transit. B1 on all 2000 held-out deployments (exact
# environment; sound speed +15 m/s; depth +2.5 m; both + 2 m hydrophone position error),
# B2 on all 800 (centre of the prior / fitted to 10 calibration fixes / true environment).
#   bash scripts/eval/mfp/run_paper.sh          # reuses the replica tables and fits in the cache
#   FRESH=1 bash scripts/eval/mfp/run_paper.sh  # rebuilds them first (focalization: hours, 40 CPU workers)
# Cache: output/eval_paper/mfp/cache; logs: output/eval_paper/mfp/logs; then collect.py
# writes output/eval_paper/mfp/results.csv and make_tables.py reads it. Needs a GPU.
# The paper's numbers come from the cached tables (built 2026-10-26/28). The simulator is
# not bit-reproducible with several processes (b1_table.py uses 10; the cached nominal B2
# table too): a rebuilt B1 table moves the exact-environment MAE by <0.1 m (3.83-3.89 m
# on 200 deployments). The 1-process paths (per-deployment B2 tables, focalization) are exact.
# Building tables also writes data/channel_option_<theta>/ (the generator's side output).
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/fer/miniconda3/envs/torch/bin/python}
M=scripts/eval/mfp; C=output/eval_paper/mfp/cache; L=output/eval_paper/mfp/logs
mkdir -p $C $L
echo "start $(date '+%F %T')"
if [ -n "$FRESH" ] || [ ! -f $C/b1_table_exact.npz ]; then
  $PY $M/b1_table.py --out b1_table_exact.npz
  $PY $M/b1_table.py --dc 15 --out b1_table_c15.npz
  $PY $M/b1_table.py --dh 2.5 --out b1_table_h25.npz
  $PY $M/b1_table.py --dc 15 --dh 2.5 --out b1_table_c15h25.npz
fi
if [ -n "$FRESH" ] || [ ! -f $C/b2_true.npz ]; then
  $PY $M/b2_tables.py --nominal
  $PY $M/b2_tables.py
fi
if [ -n "$FRESH" ] || [ ! -f $C/b2_focal.npz ]; then
  $PY $M/b2_focal.py --out b2_focal.npz > $L/b2_focal_fit.log 2>&1
fi
$PY $M/b1_mfp.py --n 2000 --table b1_table_exact.npz --tag "exact, level+phase" > $L/b1_exact.log 2>&1
$PY $M/b1_mfp.py --n 2000 --table b1_table_c15.npz --tag "sound speed +15 m/s" > $L/b1_c15.log 2>&1
$PY $M/b1_mfp.py --n 2000 --table b1_table_h25.npz --tag "depth +2.5 m" > $L/b1_h25.log 2>&1
$PY $M/b1_mfp.py --n 2000 --table b1_table_c15h25.npz --pos-err 2 --tag "compound mismatch" > $L/b1_compound.log 2>&1
echo "B1 done $(date '+%F %T')"
for e in nominal focal true; do
  $PY $M/b2_mfp.py --env $e --tag "$e env, level+phase" > $L/b2_$e.log 2>&1
done
$PY $M/collect.py
echo "=== MFP DONE $(date '+%F %T') ==="
