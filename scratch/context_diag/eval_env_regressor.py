"""Per-point errors of the true-environment regressor(s) (regressor_env.py) on the B2
held-out pool, next to the environment-blind regressor, the PFN and their fusion
(from fusion.py's npz). Usage: eval_env_regressor.py <tag>=<ckpt>[:keys] ..."""
import os, sys, numpy as np, torch as t
from types import SimpleNamespace
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scratch", "context_diag")); sys.path.insert(0, os.path.join(REPO, "scripts", "eval")); sys.path.insert(0, REPO)
import acoustic_regressor as R, regressor_env as RE
S = "/tmp/claude-1005/-home-fer-tesis/cc195f2f-8e89-4882-9745-9adf6745fe23/scratchpad/acoustic_cache_b2"
dev = t.device("cuda")
Z = np.load(os.path.join(REPO, "output/eval_paper/context_diag/b2_fusion.npz")); Y = Z["Y"]
A = SimpleNamespace(encoder="set", pos_code="raw", frame="centroid", aug="rot", d=256, layers=6, loss="nll", var_floor=0.01, feat="tone", tokenizer="mlp", no_ffn=False, summary="readout", pos_err=0.0)
z = np.load(os.path.join(S, "train.npz")); la = t.as_tensor(z["tone"]).abs().clamp_min(1e-12).log(); st = (float(la.mean()), float(la.std()))
P0 = [R.load_pool(S, p, "tone", dev, st) for p in ("val", "test")]

@t.no_grad()
def reg(ck, keys):
    T = RE.env_table(keys, False); mu, var = [], []
    net = R.Net(A, 3 + len(keys), 10).to(dev); net.load_state_dict(t.load(ck, map_location=dev)); net.eval()
    for D in P0:
        e = t.as_tensor(T[D["gid"]], device=dev)[:, None, None, :].expand(D["N"], 50, 10, len(keys)).reshape(-1, 10, len(keys))
        Dk = dict(D, F=t.cat([D["F"], e], -1))
        for a in range(0, D["N"] * 50, 8192):
            idx = t.arange(a, min(a + 8192, D["N"] * 50), device=dev)
            pos, F, _, c = R.batch(Dk, idx, A, train=False)
            o = net(pos, F)                            # float32
            m, v = R.gaussian(o, A); mu.append((m * 100 + c).cpu()); var.append((v * 1e4).cpu())
    return t.cat(mu).view(-1, 50, 2).numpy(), t.cat(var).view(-1, 50, 2).numpy()

sp = np.load(os.path.join(S, "val.npz"))["sp"][0]; r = np.linalg.norm(Y - sp.mean(0), axis=-1)
bins = ((0, 100), (100, 200), (200, 300), (300, 400), (400, 9999))
print(f"{'':44s}  MAE  median   p90   p99 late40 | by range from the centroid " + " ".join(f"{lo}-{hi}" for lo, hi in bins))
def show(n, m):
    e = np.linalg.norm(m - Y, axis=-1)
    print(f"  {n:42s} {e.mean():5.2f} {np.median(e):5.2f} {np.percentile(e, 90):6.2f} {np.percentile(e, 99):6.2f} {e[:, 10:].mean():5.2f} | "
          + "  ".join(f"{e[(r >= lo) & (r < hi)].mean():6.2f}" for lo, hi in bins), flush=True)
f = lambda ms, vs: sum(m / v for m, v in zip(ms, vs)) / sum(1 / v for v in vs)
show("regressor, environment-blind", Z["reg_mu"])
for s in ("s0", "s1"):
    show(f"PFN {s}, 10 context points", Z[f"{s}_n10_mu"])
    show(f"PFN {s} + regressor fused", f([Z[f"{s}_n10_mu"], Z["reg_mu"]], [Z[f"{s}_n10_var"], Z["reg_var"]]))
out = {}
for spec in sys.argv[1:]:
    tag, rest = spec.split("=", 1); ck, _, keys = rest.partition(":")
    m, v = reg(ck, (keys or "h0,hr0,c,c2f,k").split(",")); out[tag + "_mu"], out[tag + "_var"] = m, v
    show(f"regressor given {keys or 'the true environment'} ({tag})", m)
    show(f"  + PFN s0 fused", f([m, Z["s0_n10_mu"]], [v, Z["s0_n10_var"]]))
np.savez_compressed(os.path.join(REPO, "output/eval_paper/context_diag/b2_env_regressor.npz"), Y=Y, **out)
