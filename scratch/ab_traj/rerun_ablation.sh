#!/usr/bin/env bash
# s4_ablation with tiers 1-3.
#
# Tier 4 (permutation + sensor_dropout) needs `sensor_mask`, which only the
# TOKENIZED SpatialEncoder supports. The whole point of this ablation is that one
# of its arms is `no_tokenize` -- the flat control -- so tier 4 can never apply to
# it. Same cause as the B3 failure; I should have carried the fix across when I
# wrote the chain.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
SH_DIR=data/random_task/hermite_shared/processed/geometry_split
M=""
for pair in "full:paper_spatial_cnp_full" "no_tokenize:paper_abl_notokenize" \
            "no_position:paper_abl_nopos" "no_attention:paper_abl_noattn" \
            "mean_pool:paper_abl_meanpool"; do
  lbl=${pair%%:*}; exp=${pair##*:}
  for c in $($PY - "$exp" <<'EOF'
import glob,re,sys
best={}
for p in glob.glob(f"output/hydra/training/*/{sys.argv[1]}/seed_*/best.pt"):
    s=int(p.split("/seed_")[1].split("/")[0]); d=re.search(r"/training/([\d-]+)/",p).group(1)
    if s not in best or d>best[s][0]: best[s]=(d,p)
print("\n".join(p for _,p in sorted(best.values())))
EOF
  ); do M="$M $lbl=$c"; done
done
echo "=== s4_ablation rerun, tiers 1-3  $(date '+%F %T') ==="
$PY -u scripts/eval/eval_suite.py --models $M --tiers 1,2,3 \
    --out-dir output/eval_paper/s4_ablation --data-dir $SH_DIR \
    > output/eval_paper/s4_ablation.log 2>&1
echo "  rc=$?  $(date '+%T')"
tail -12 output/eval_paper/s4_ablation.log | sed 's/^/  /'
