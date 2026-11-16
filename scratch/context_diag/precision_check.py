"""Evaluation-only: does bfloat16 rounding of the outputs cost accuracy?
For each checkpoint, calibration context of 10 points on B2, three variants:
  bf16      as trained and evaluated so far (the head runs inside the autocast)
  head32    encoders in bf16, the output head in float32
  fp32      everything in float32
"""
import sys, numpy as np, torch as t, torch.nn as nn
REPO = "/home/fer/tesis/sensor-agnostic-underwater-localization"
sys.path.insert(0, REPO + "/scratch/context_diag"); sys.path.insert(0, REPO + "/scripts/eval"); sys.path.insert(0, REPO)
import ctx_controls as C
E = C.E

class Head32(nn.Module):
    def __init__(self, head): super().__init__(); self.head = head
    def forward(self, h):
        with t.autocast("cuda", enabled=False):
            return self.head(h.float())

def find(model, cls_name):
    for m in model.modules():
        if type(m).__name__ == cls_name: return m
    return None

dev = t.device("cuda")
D = C.load(sys.argv[1], ["val", "test"], 0)
own, foreign = C.donors(D)
Y = D["Y"][..., :2].numpy()
r = np.linalg.norm(Y - D["SP"].numpy()[:, :, :2].mean(1)[:, None], axis=-1); bins = [0, 100, 200, 300, 400, 1e9]
for spec in sys.argv[2:]:
    lbl, path = spec.split("=", 1)
    for variant in ("bf16", "head32", "fp32"):
        b = E.load_model(path, dev)
        cr = find(b["model"], "ContextRegressor")
        if cr is None:
            if variant == "head32": continue
            if variant == "fp32":
                for m in b["model"].modules():
                    if hasattr(m, "bf16"): m.bf16 = False
        elif variant == "head32": cr.head = Head32(cr.head)
        elif variant == "fp32": cr.bf16 = False
        mu, score = C.run_pfn(b, D, "calib_n10", own, foreign, dev)
        e = np.linalg.norm(mu - Y, axis=-1)
        print(f"{lbl:10s} {variant:7s} MAE {e.mean():.3f} median {np.median(e):.3f} by range",
              [round(float(e[(r >= lo) & (r < hi)].mean()), 2) for lo, hi in zip(bins[:-1], bins[1:])], flush=True)
        del b
