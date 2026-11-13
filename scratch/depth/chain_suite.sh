#!/usr/bin/env bash
# Eval suite over the 24 depth-stratified checkpoints, once training ends.
#
# One group per training arm, so the output tables line up with b2_prefix (same
# model labels, same data, same protocol) and the three arms can be read as one
# row each. The suite is run on pathD's held-out pool, which is IDENTICAL across
# arms -- only what the models were trained on differs.
#
# Tiers 1,2,3 only: tier 4 needs sensor_mask, and we have no reason to spend the
# extra time on permutation/dropout here. Region MAE (tier 1) is the number the
# whole study is about.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
LAUNCHER=${1:-}

if [ -n "$LAUNCHER" ]; then
  echo "=== waiting for training (pid $LAUNCHER) — $(date '+%F %T') ==="
  while kill -0 "$LAUNCHER" 2>/dev/null; do sleep 60; done
  echo "=== training finished $(date '+%F %T') ==="
fi

pick () {   # newest checkpoint per seed, only runs that converged
  $PY - "$1" <<'EOF'
import glob,re,sys,os
exp=sys.argv[1]; best={}
for p in glob.glob(f"output/hydra/training/*/{exp}/seed_*/best.pt"):
    s=int(p.split("/seed_")[1].split("/")[0])
    d=re.search(r"/training/([\d-]+)/",p).group(1)
    lg=f"scratch/depth/{exp}_s{s}.log"
    if not (os.path.exists(lg) and "done. best val mae" in open(lg).read()): continue
    if s not in best or d>best[s][0]: best[s]=(d,p)
print("\n".join(p for _,p in sorted(best.values())))
EOF
}

for ARM in shallow deep; do
  M=""
  for pair in "cnp:DS_${ARM}_cnp" "cnp_ffn:DS_${ARM}_cnpF" "pfn:DS_${ARM}_pfn" "rcnp:DS_${ARM}_rcnp"; do
    lbl=${pair%%:*}; exp=${pair##*:}
    for c in $(pick "$exp"); do M="$M $lbl=$c"; done
  done
  n=$(echo $M | wc -w)
  echo ""
  echo "=== suite: ds_${ARM}  ($n checkpoints)  $(date '+%F %T') ==="
  $PY -u scripts/eval/eval_suite.py --models $M --tiers 1,2,3 \
      --out-dir output/eval_paper/ds_${ARM} \
      > output/eval_paper/ds_${ARM}.log 2>&1
  echo "  rc=$?  $(date '+%T')"
done

echo ""
echo "=========== DEPTH STUDY: TEST-REGION MAE ==========="
$PY - <<'EOF'
import csv, os
def G(g,m,k):
    f=f"output/eval_paper/{g}/results.csv"
    if not os.path.exists(f): return None
    r=next((x for x in csv.DictReader(open(f)) if x["model"]==m
            and x["test"]=="region_mae" and x["metric"]==k), None)
    return (float(r["mean"]),float(r["std"])) if r else None
print(f"\n  {'model':<9}{'arm':<10}{'overall':>16}{'shallow':>10}{'deep':>10}{'deep/shal':>11}")
for m in ("pfn","rcnp","cnp_ffn","cnp"):
    for arm,g in (("full","b2_prefix"),("shallow-only","ds_shallow"),("deep-only","ds_deep")):
        o,s,d = G(g,m,"overall"), G(g,m,"shallow"), G(g,m,"deep")
        if not o: continue
        ratio = f"{d[0]/s[0]:.2f}x" if (s and d) else "-"
        print(f"  {m if arm=='full' else '':<9}{arm:<10}{f'{o[0]:.2f} ± {o[1]:.2f}':>16}"
              f"{s[0] if s else float('nan'):>10.2f}{d[0] if d else float('nan'):>10.2f}{ratio:>11}")
    print()
print("  The question: does deep-only close the deep-bucket gap relative to full,")
print("  and does shallow-only degrade by the same factor (= pure size effect)?")
EOF
echo "=== CHAIN COMPLETE $(date '+%F %T') ==="
