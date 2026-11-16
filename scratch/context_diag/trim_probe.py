"""Evaluation-only probes of the attention model's components (B2, 10 calibration
points, all 50 points, float32), to see where it could be trimmed. No retraining:
a component whose removal costs nothing is unused; a component whose removal
costs a lot may still be replaceable in a retrained, smaller model.

    python scratch/context_diag/trim_probe.py label=<best.pt> ...
"""
import os, sys, copy
import numpy as np
import torch as t
import torch.nn as nn
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scratch", "context_diag")); sys.path.insert(0, os.path.join(REPO, "scripts", "eval")); sys.path.insert(0, REPO)
import ctx_controls as C            # noqa: E402

dev = t.device("cuda")
D = C.load(os.path.join(REPO, "data/pathD/hermite_distinct/processed/geometry_split"), ["val", "test"], 0)
own, foreign = C.donors(D)
Y = D["Y"][..., :2].numpy()


def find(model):
    return next(m for m in model.modules() if type(m).__name__ == "ContextRegressor")


class Zero(nn.Module):
    def forward(self, q, keys, pad=None):
        return t.zeros_like(q)


class Skip(nn.Module):
    def forward(self, c, src_key_padding_mask=None):
        return c


def mae(b, cond="calib_n10"):
    mu, _ = C.run_pfn(b, D, cond, own, foreign, dev)
    return float(np.linalg.norm(mu - Y, axis=-1).mean())


for spec in sys.argv[1:]:
    lbl, path = spec.split("=", 1)
    base = C.E.load_model(path, dev)
    L = len(find(base["model"]).ctx_attn)
    print(f"== {lbl}: intact {mae(base):.2f}", flush=True)

    def variant(edit):
        b = dict(base, model=copy.deepcopy(base["model"])); edit(find(b["model"])); return b

    # 1. Cross-attention blocks (one before each of the target encoder's layers).
    def keep(layers):
        def edit(cr):
            for i in range(L):
                if i not in layers: cr.ctx_attn[str(i)] = Zero()
        return edit
    print("  drop one block      " + "  ".join(f"L{i + 1} {mae(variant(keep(set(range(L)) - {i}))):.2f}" for i in range(L)), flush=True)
    print("  keep one block      " + "  ".join(f"L{i + 1} {mae(variant(keep({i}))):.2f}" for i in range(L)), flush=True)
    print("  keep the first k    " + "  ".join(f"k={k} {mae(variant(keep(set(range(k))))):.2f}" for k in range(1, L)), flush=True)
    print("  keep the last k     " + "  ".join(f"k={k} {mae(variant(keep(set(range(L - k, L))))):.2f}" for k in range(1, L)), flush=True)

    # 2. Transformer over the context points (2 layers).
    def pool(n):
        def edit(cr):
            if n == 0: cr.ctx_pool = Skip()
            else: cr.ctx_pool.layers = nn.ModuleList(list(cr.ctx_pool.layers)[:n])
        return edit
    print(f"  context transformer: 0 layers {mae(variant(pool(0))):.2f}   first layer only {mae(variant(pool(1))):.2f}", flush=True)

    # 3. Context points' sensor encoder truncated to its first k layers.
    def ctx_depth(k):
        def edit(cr): cr.ctx_encoder.layers.layers = nn.ModuleList(list(cr.ctx_encoder.layers.layers)[:k])
        return edit
    print("  context encoder, first k layers  " + "  ".join(f"k={k} {mae(variant(ctx_depth(k))):.2f}" for k in (5, 4, 3, 2)), flush=True)

    # 4. Context encoder's layers replaced by the target encoder's (own input map kept).
    def tie(cr): cr.ctx_encoder.layers = cr.spatial_encoder.layers
    print(f"  context encoder using the target encoder's layers {mae(variant(tie)):.2f}", flush=True)
    del base
