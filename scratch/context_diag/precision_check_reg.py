"""Evaluation-only: the no-context regressor on B2 with and without bfloat16."""
import sys, contextlib, numpy as np, torch as t
REPO = "/home/fer/tesis/sensor-agnostic-underwater-localization"
sys.path.insert(0, REPO + "/scratch/context_diag"); sys.path.insert(0, REPO + "/scripts/eval"); sys.path.insert(0, REPO)
import ctx_controls as C
dev = t.device("cuda")
DD = REPO + "/data/pathD/hermite_distinct/processed/geometry_split"
CACHE = "/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache_b2"
D = C.load(DD, ["val", "test"], 0)
Y = D["Y"][..., :2].numpy()
real = t.autocast
class Off(real):
    def __init__(self, *a, **k): k["enabled"] = False; super().__init__(*a, **k)
for lbl, ck, feat in [("tone s0", "acoustic_tone_nll_drop.pt", "tone"), ("tone s1", "acoustic_tone_nll_drop_s1.pt", "tone"),
                      ("raw", "acoustic_raw_nll_drop.pt", "raw")]:
    for variant in ("bf16", "fp32"):
        t.autocast = real if variant == "bf16" else Off
        try:
            mu = C.run_reg(REPO + "/output/eval_paper/baselines/b2/" + ck, CACHE, D, ["val", "test"], feat, "nll", dev)
        finally:
            t.autocast = real
        e = np.linalg.norm(mu - Y, axis=-1)
        print(f"{lbl:8s} {variant} MAE {e.mean():.3f} median {np.median(e):.3f}", flush=True)
