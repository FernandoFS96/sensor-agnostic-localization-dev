"""Test-only re-score of every evaluation group behind the AOR paper (2026-11-11).

Until now each score pooled the validation and test halves of the held-out set,
and the validation half also selected the checkpoint. This chain re-runs the same
evaluations on the test half only (EVAL_POOLS=test) into output/eval_paper_test/,
leaving output/eval_paper/ untouched.

Checkpoint selection is the one the original chains used (newest date directory
per seed; `require_done` where the original required a finished log), so only the
held-out pool changes. Step 0 proves it: s4_ablation is re-run on the POOLED set
and must reproduce the published region MAE before anything else runs.

Tiers are those the paper reads: 1-2 for the tables (region MAE, noise/zero,
order), 3-4 only where a figure or the text uses them (B1 curves: b1_prefix and
b1_ranp; s4_ablation's fix placement in Appendix A; the development-set groups
keep their original tiers because they cost minutes).

The S-RecCNP runs trained by train_srec.sh join s4_family (shared) and
ab_distinct (distinct) once their training finishes; those two groups are also
scored on the pooled set into *_srec groups, so either protocol has the new rows.
"""
import csv, glob, os, re, subprocess, sys, time, collections

REPO = "/home/fer/tesis/sensor-agnostic-underwater-localization"
PY = "/home/fer/miniconda3/envs/torch/bin/python"
os.chdir(REPO)
TEST_OUT, POOL_OUT = "output/eval_paper_test", "output/eval_paper"
LOG = "scratch/rescore"
SH = "data/random_task/hermite_shared/processed/geometry_split"
DI = "data/random_task/hermite_distinct/processed_common/geometry_split"
os.makedirs(TEST_OUT, exist_ok=True)


def say(*a):
    print(*a, flush=True)


def ckpts(exp, done_logs=None):
    """Newest date directory per seed; with done_logs, only seeds whose training
    log (in any of those directories) says 'done. best val mae'."""
    best = {}
    for p in glob.glob(f"output/hydra/training/*/{exp}/seed_*/best.pt"):
        s = int(p.split("/seed_")[1].split("/")[0])
        d = re.search(r"/training/([\d-]+)/", p).group(1)
        if s not in best or d > best[s][0]:
            best[s] = (d, p)
    if done_logs:
        keep = {}
        for s, v in best.items():
            ok = any(os.path.exists(f"{dl}/{exp}_s{s}.log") and
                     "done. best val mae" in open(f"{dl}/{exp}_s{s}.log").read()
                     for dl in done_logs)
            if ok:
                keep[s] = v
            else:
                say(f"    [skipped] {exp} s{s}: no finished log")
        best = keep
    return [p for _, p in sorted(best.values())]


def suite(group, mods, ddir=None, tiers="1,2", pools="test", out_root=TEST_OUT):
    args = []
    for lbl, (exp, dl) in mods.items():
        for p in ckpts(exp, dl):
            args.append(f"{lbl}={p}")
    if not args:
        say(f"  !! {group}: no checkpoints"); return False
    od = f"{out_root}/{group}"
    cmd = [PY, "-u", "scripts/eval/eval_suite.py", "--models", *args,
           "--tiers", tiers, "--out-dir", od]
    if ddir:
        cmd += ["--data-dir", ddir]
    n = collections.Counter(a.split("=")[0] for a in args)
    say(f"\n--- {group} [{pools}] {dict(n)} tiers={tiers}  {time.strftime('%F %T')}")
    env = dict(os.environ, EVAL_POOLS=pools, TQDM_DISABLE="1")
    with open(f"{od}.log", "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    say(f"    rc={rc}  {time.strftime('%F %T')}")
    if rc:
        for ln in open(f"{od}.log").read().splitlines()[-10:]:
            say(f"      {ln}")
    return rc == 0


def script(name, cmd, out):
    say(f"\n--- {name} [test]  {time.strftime('%F %T')}")
    env = dict(os.environ, EVAL_POOLS="test", TQDM_DISABLE="1")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(f"{out}.log", "w") as lf:
        rc = subprocess.call([PY, "-u", *cmd], stdout=lf, stderr=subprocess.STDOUT, env=env)
    say(f"    rc={rc}  {time.strftime('%F %T')}")


def overall(path, model):
    for r in csv.DictReader(open(path)):
        if r["model"] == model and r["test"] == "region_mae" and r["metric"] == "overall":
            return float(r["mean"])


def wait_for(flag):
    say(f"\n=== waiting for {flag}  {time.strftime('%F %T')}")
    while not os.path.exists(f"{LOG}/{flag}"):
        time.sleep(300)
    say(f"=== {flag} present  {time.strftime('%F %T')}")


# per-label (experiment, directories holding finished training logs or None)
FAMILY = {"flat_cnp": ("paper_flat_cnp", None), "flat_anp": ("paper_flat_anp", None),
          "spatial_cnp": ("paper_spatial_cnp_full", None), "spatial_anp": ("paper_spatial_anp", None),
          "rcnp": ("paper_rcnp", None), "ranp": ("paper_ranp", None),
          "spatial_ranp": ("paper_spatial_ranp", None)}
ABLATION = {"full": ("paper_spatial_cnp_full", None), "no_tokenize": ("paper_abl_notokenize", None),
            "no_position": ("paper_abl_nopos", None), "no_attention": ("paper_abl_noattn", None),
            "mean_pool": ("paper_abl_meanpool", None)}
AB_DONE = ["scratch/ab_traj"]
DS_DONE = ["scratch/depth"]
NEW_DONE = [LOG]

say(f"=== TEST-ONLY RE-SCORE  {time.strftime('%F %T')} ===")

# ---- 0. reproduction check on the pooled set ---------------------------------
chk = f"{LOG}/check_pooled"
os.makedirs(chk, exist_ok=True)
if not suite("s4_ablation", ABLATION, SH, tiers="1", pools="val,test", out_root=chk):
    sys.exit("reproduction check failed to run")
bad = []
for m in ABLATION:
    a = overall(f"{chk}/s4_ablation/results.csv", m)
    b = overall(f"{POOL_OUT}/s4_ablation/results.csv", m)
    say(f"    check {m:<14} new {a:.2f}  published {b:.2f}")
    if a is None or b is None or abs(a - b) > 0.05:
        bad.append(m)
if bad:
    sys.exit(f"reproduction check FAILED for {bad}: not re-scoring")
say("    reproduction check passed")

# ---- 1. development set, B1, B2, depth arms (existing checkpoints) -----------
suite("s4_ablation", ABLATION, SH, tiers="1,2,3")
script("latent_gap_v2", ["scripts/eval/latent_gap.py", "--out-dir", f"{TEST_OUT}/latent_gap_v2"],
       f"{TEST_OUT}/latent_gap_v2")
for name, dd in (("dev_shared", SH), ("dev_distinct", DI),
                 ("b1", "data/pfn10kd/hermite_distinct/processed/geometry_split"),
                 ("b2", "data/pathD/hermite_distinct/processed/geometry_split")):
    script(f"baselines/{name}", ["scripts/eval/trivial_baselines.py", "--data-dir", dd,
                                 "--out-dir", f"{TEST_OUT}/baselines/{name}"],
           f"{TEST_OUT}/baselines/{name}")
suite("b1_prefix", {"cnp": ("CL_A_cnp", None), "cnp_ffn": ("CL_A_cnpF", None),
                    "pfn": ("CL_A_pfn", None), "rcnp": ("CL_A_rcnp", None)}, tiers="1,2,3,4")
suite("b1_ranp", {"ranp": ("CL_A_ranp", None)}, tiers="1,2,3,4")
suite("b1_calib", {"cnp": ("CL_B_cnp", None), "cnp_ffn": ("CL_B_cnpF", None),
                   "pfn": ("CL_B_pfn", None)})
suite("b2_prefix", {"cnp": ("PD_A_cnp", None), "cnp_ffn": ("PD_A_cnpF", None),
                    "pfn": ("PD_A_pfn", None), "rcnp": ("PD_A_rcnp", None)})
suite("b2_calib", {"cnp": ("PD_B_cnp", None), "cnp_ffn": ("PD_B_cnpF", None),
                   "pfn": ("PD_B_pfn", None)})
for arm in ("shallow", "deep"):
    suite(f"ds_{arm}", {"cnp": (f"DS_{arm}_cnp", DS_DONE), "cnp_ffn": (f"DS_{arm}_cnpF", DS_DONE),
                        "pfn": (f"DS_{arm}_pfn", DS_DONE), "rcnp": (f"DS_{arm}_rcnp", DS_DONE)})
say(f"\n=== existing checkpoints re-scored  {time.strftime('%F %T')} ===")

# ---- 2. shared variant with the new S-RecCNP ----------------------------------
wait_for("SHARED_DONE")
fam = dict(FAMILY, spatial_rcnp=("paper_spatial_rcnp", NEW_DONE))
suite("s4_family", fam, SH, tiers="1,2,3,4")
suite("s4_family_srec", fam, SH, tiers="1,2,3,4", pools="val,test", out_root=POOL_OUT)

# ---- 3. distinct variant with the new S-RecCNP --------------------------------
wait_for("DISTINCT_DONE")
ab = {"spatial_cnp": ("AB_dist_spatial_cnp", AB_DONE), "rcnp": ("AB_dist_rcnp", AB_DONE),
      "spatial_rcnp": ("AB_dist_spatial_rcnp", NEW_DONE)}
suite("ab_distinct", ab, DI, tiers="1,2,3,4")
suite("ab_distinct_srec", ab, DI, tiers="1,2,3,4", pools="val,test", out_root=POOL_OUT)
say(f"\n=== RE-SCORE COMPLETE  {time.strftime('%F %T')} ===")
