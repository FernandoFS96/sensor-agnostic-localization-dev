"""Precision-weighted fusion of the PFN (calibration context) and the no-context
regressor on B2, per axis, with each model's own predicted variance. Eval only."""
import os, sys, numpy as np, torch as t
from types import SimpleNamespace
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scratch", "context_diag"))
import ctx_controls as C
R, E = C.R, C.E
S = "/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache_b2"
DD = os.path.join(REPO, "data/pathD/hermite_distinct/processed/geometry_split")
dev = t.device("cuda")
D = C.load(DD, ["val", "test"], 0); own, foreign = C.donors(D); Y = D["Y"][..., :2].numpy()

@t.no_grad()
def pfn(path, k):
    b = E.load_model(path, dev); ym, ys = b["ym"], b["ys"]; mu, var = [], []
    for a in range(0, len(Y), 128):
        idx = np.arange(a, min(a + 128, len(Y)))
        cx, cy, _ = C.context(D, f"calib_n{k}", idx, own, foreign, dev)
        m, v = b["model"](cx, (cy - ym) / ys, D["X"][idx].to(dev), None, sensor_pos=D["SP"][idx].to(dev))[:2]
        mu.append((m * ys + ym)[..., :2].cpu()); var.append((v * ys ** 2)[..., :2].cpu())
    return t.cat(mu).numpy(), t.cat(var).numpy()

@t.no_grad()
def reg():
    A = SimpleNamespace(encoder="set", pos_code="raw", frame="centroid", aug="rot", d=256, layers=6, loss="nll",
                        var_floor=0.01, feat="tone", tokenizer="mlp", no_ffn=False, summary="readout", pos_err=0.0)
    z = np.load(os.path.join(S, "train.npz")); la = t.as_tensor(z["tone"]).abs().clamp_min(1e-12).log()
    st = (float(la.mean()), float(la.std())); mu, var = [], []
    net = R.Net(A, 3, 10).to(dev); net.load_state_dict(t.load(os.path.join(REPO, "output/eval_paper/baselines/b2/acoustic_tone_nll_drop.pt"), map_location=dev)); net.eval()
    for p in ("val", "test"):
        P = R.load_pool(S, p, "tone", dev, st)
        for a in range(0, P["N"] * 50, 8192):
            idx = t.arange(a, min(a + 8192, P["N"] * 50), device=dev)
            pos, F, _, c = R.batch(P, idx, A, train=False)
            o = net(pos, F)                            # float32
            m, v = R.gaussian(o, A); mu.append((m * 100 + c).cpu()); var.append((v * 1e4).cpu())
    return t.cat(mu).view(-1, 50, 2).numpy(), t.cat(var).view(-1, 50, 2).numpy()

mr, vr = reg(); assert len(mr) == len(Y)
err = lambda m: np.linalg.norm(m - Y, axis=-1)
def show(n, m): e = err(m); print(f"  {n:46s} MAE {e.mean():5.2f}  median {np.median(e):5.2f}  p90 {np.percentile(e, 90):5.2f}  late40 {e[:, 10:].mean():5.2f}", flush=True)
show("regressor (no context)", mr)
runs = {"s0": "output/hydra/training/2026-10-29/PD_B_pfnset_fixdrop/seed_0/best.pt", "s1": "output/hydra/training/2026-11-05/PD_B_pfnset_fixdrop/seed_1/best.pt"}
out = dict(Y=Y, reg_mu=mr, reg_var=vr)
for s, p in runs.items():
    for k in (1, 10):
        mp, vp = pfn(os.path.join(REPO, p), k); out[f"{s}_n{k}_mu"], out[f"{s}_n{k}_var"] = mp, vp
        show(f"PFN {s}, {k} context points", mp)
        w = (1 / vp) / (1 / vp + 1 / vr)
        show(f"  fused with the regressor (precision weights)", w * mp + (1 - w) * mr)
        sel = (vp.sum(-1) < vr.sum(-1))[..., None]
        show(f"  pick the model with the smaller variance", np.where(sel, mp, mr))
        print(f"    PFN picked on {sel.mean():.2f} of the points; mean PFN weight {w.mean():.2f}")
w0, w1 = 1 / out["s0_n10_var"], 1 / out["s1_n10_var"]
show("two PFN seeds fused (ensemble control)", (w0 * out["s0_n10_mu"] + w1 * out["s1_n10_mu"]) / (w0 + w1))
wr = 1 / vr
show("two PFN seeds and the regressor fused", (w0 * out["s0_n10_mu"] + w1 * out["s1_n10_mu"] + wr * mr) / (w0 + w1 + wr))
np.savez_compressed(os.path.join(REPO, "output/eval_paper/context_diag/b2_fusion.npz"), **out)
