#!/usr/bin/env bash
# 3-seed evaluation of the paper sweep. Auto-discovers every seed checkpoint per
# model and passes them as repeated label=path to eval_suite (which aggregates
# mean +/- std over seeds). Run AFTER run_paper_sweep.sh has trained seeds 0,1,2.
#   bash scripts/eval/run_eval_3seed.sh
cd "$(dirname "$0")/../.."
PY=/home/fer/miniconda3/envs/torch/bin/python
# echo "  label=path" for every seed checkpoint of paper_<exp>
S(){ for p in $(ls output/hydra/training/*/paper_$2/seed_*/best.pt 2>/dev/null); do printf " %s=%s" "$1" "$p"; done; }
OUT=output/eval_suite_3seed; mkdir -p "$OUT"

echo "===== HERMITE MAIN (tiers 1-4) $(date) ====="
$PY scripts/eval/eval_suite.py --tiers 1,2,3,4 --cap-traj 6 --out-dir "$OUT/hermite_main" --models \
  $(S flat_cnp flat_cnp) $(S flat_anp flat_anp) \
  $(S spatial_cnp spatial_cnp_full) $(S spatial_anp spatial_anp) \
  $(S rcnp rcnp) $(S ranp ranp) $(S spatial_ranp spatial_ranp) \
  > "$OUT/hermite_main.log" 2>&1
echo "main done $(date)"

echo "===== HERMITE ABLATION (tiers 1,2) $(date) ====="
$PY scripts/eval/eval_suite.py --tiers 1,2 --cap-traj 6 --out-dir "$OUT/hermite_ablation" --models \
  $(S full spatial_cnp_full) $(S no_tokenize abl_notokenize) $(S no_position abl_nopos) \
  $(S no_attention abl_noattn) $(S mean_pool abl_meanpool) \
  > "$OUT/hermite_ablation.log" 2>&1
echo "ablation done $(date)"

echo "===== SPIRAL cross-dataset (tiers 1,2) $(date) ====="
$PY scripts/eval/eval_suite.py --tiers 1,2 --cap-traj 6 --out-dir "$OUT/spiral" --models \
  $(S flat_cnp flat_cnp_spiral) $(S flat_anp flat_anp_spiral) $(S spatial_cnp spatial_cnp_spiral) \
  > "$OUT/spiral.log" 2>&1
echo "spiral done $(date)"
echo "ALL EVAL DONE $(date)"
