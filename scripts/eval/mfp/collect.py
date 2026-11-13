"""MFP logs -> output/eval_paper/mfp/results.csv (bench, variant, overall, interp, extrap,
shallow, deep), the paper protocol: targets = points 11-50 of every held-out transit.
Read by paper/scripts/make_tables.py (Table tab:mfp).
    python scripts/eval/mfp/collect.py [--logs output/eval_paper/mfp/logs]"""
import argparse, ast, csv, os, re
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(REPO, "output", "eval_paper", "mfp")
RUNS = [("B1", "exact", "b1_exact.log"), ("B1", "c15", "b1_c15.log"),
        ("B1", "h25", "b1_h25.log"), ("B1", "compound", "b1_compound.log"),
        ("B2", "nominal", "b2_nominal.log"), ("B2", "focal", "b2_focal.log"),
        ("B2", "true", "b2_true.log")]
ap = argparse.ArgumentParser(); ap.add_argument("--logs", default=os.path.join(OUT, "logs"))
A = ap.parse_args()
rows = []
for bench, var, log in RUNS:
    p = os.path.join(A.logs, log)
    if not os.path.exists(p):
        print("missing", log); continue
    txt = open(p).read()
    if bench == "B1":
        m = re.search(r"MFP on (\d+) B1 deployments, (\d+) transits: MAE ([\d.]+) m", txt)
        reg = ast.literal_eval(re.search(r"by region: (\{.*\})", txt).group(1))
        overall = float(m.group(3))
    else:
        m = re.search(r"MFP on (\d+) B2 deployments, (\d+) transits: .*points 11-50 ([\d.]+) m", txt)
        reg = ast.literal_eval(re.search(r"by region, points 11-50: (\{.*\})", txt).group(1))
        overall = float(m.group(3))
    rows.append(dict(bench=bench, variant=var, deployments=m.group(1), transits=m.group(2),
                     overall=f"{overall:.4f}", **{k: f"{v:.4f}" for k, v in reg.items()}))
    print(rows[-1])
os.makedirs(OUT, exist_ok=True)
keys = ["bench", "variant", "deployments", "transits", "overall", "interp", "extrap", "shallow", "deep"]
with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(rows)
print("wrote", os.path.join(OUT, "results.csv"))
