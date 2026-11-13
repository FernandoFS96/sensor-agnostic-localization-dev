"""Matched-field processing on B1 with the true (fixed) environment, or a mismatched one.

Per target point: the measured complex tone amplitude at each of the 10 sensors is
compared with the replica table r(d) at every candidate source position p, where
d_i(p) = |p - s_i| (3-D, source at z = 0). Robust (Cauchy) misfit on log amplitude
and wrapped phase, summed over sensors; exhaustive 2 m grid over the array centroid
+- 1100 m, then a 0.25 m refinement. No context, no trajectory: acoustics + array only.
Scored on eval_suite's transits and targets (points 11-50).

    python scripts/eval/mfp/b1_mfp.py --n 2000 --table b1_table_exact.npz --tag "exact, level+phase"
    python scripts/eval/mfp/b1_mfp.py --n 2000 --table b1_table_c15h25.npz --pos-err 2 --tag "compound mismatch"
"""
import os, sys, argparse, pickle, time, numpy as np, torch as t
from collections import defaultdict
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(REPO, "output", "eval_paper", "mfp", "cache")
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
import eval_suite as E

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--data-dir", default=os.path.join(REPO, "data/pfn10kd/hermite_distinct/processed/geometry_split"))
ap.add_argument("--work", default=WORK)
ap.add_argument("--table", default="b1_table_exact.npz", help="b1_table.py output, in --work or a path")
ap.add_argument("--theta", default="known", help="known | 0.1 | 0.2: which replica table each transit uses")
ap.add_argument("--proc", default="cplx", help="cplx (level+phase) | level | level_demean")
ap.add_argument("--pos-err", type=float, default=0.0, help="sd [m] of hydrophone position error in the replicas")
ap.add_argument("--tag", default="")
A = ap.parse_args(); N_DEPLOY = A.n
dev = t.device("cuda")
T = np.load(os.path.join(A.work, A.table))
DG = np.arange(1.0, 1300.0, 0.05)
TAB = {th: t.tensor(np.interp(DG, T[f"d_{th}"], T[f"z_{th}"].real)
                    + 1j * np.interp(DG, T[f"d_{th}"], T[f"z_{th}"].imag), dtype=t.complex64, device=dev)
       for th in (0.1, 0.2)}
SIG = {0.1: 0.05, 0.2: 0.10}                          # robust scale, from the table validation
w0 = np.pi / 20; nn_ = np.arange(201)
CS = t.tensor(np.stack([np.cos(w0 * nn_), np.sin(w0 * nn_)]), dtype=t.float32, device=dev)

def tone(X):                                          # (P, 2010) -> (P, 10) complex
    X = X.view(-1, 201, 10)
    return t.complex(t.einsum("pns,n->ps", X, CS[0]), -t.einsum("pns,n->ps", X, CS[1])) * 2 / 201

def replica(th, d):                                   # d (...,) metres -> complex replica
    idx = ((d - 1.0) / 0.05).clamp(0, len(DG) - 2)
    i0 = idx.floor().long(); f = (idx - i0).to(t.float32)
    tab = TAB[th]
    return tab[i0] * (1 - f) + tab[i0 + 1] * f

def misfit(z, r, th):                                 # z (B, 1, S), r (1|B, C, S) -> (B, C)
    e_a = t.log(r.abs() + 1e-12) - t.log(z.abs() + 1e-12)
    if A.proc == "level_demean":                      # no source-level knowledge (RANP App. O)
        e_a = e_a - e_a.mean(-1, keepdim=True)
    e2 = e_a ** 2 if A.proc != "cplx" else e_a ** 2 + t.angle(r * z.conj()) ** 2
    return t.log1p(e2 / SIG[th] ** 2).sum(-1)

data = E.load_data(A.data_dir, t.device("cpu"), load_cap=18)
items = E.subsample_test(data["by_gid"], 6, 0)
by = defaultdict(list)
for it in items: by[it[2]].append(it)
gids = sorted(by)[:N_DEPLOY]
# theta of each transit: eval items do not carry it, so recover it from the pickles' order
theta_of = {}
for pool in ("val", "test"):
    for smp in pickle.load(open(os.path.join(A.data_dir, f"{pool}_data.pkl"), "rb")):
        theta_of[(int(smp["geometry_id"]), np.asarray(smp["X"], np.float32)[0, :5].tobytes())] = round(float(smp["theta"]), 1)

g1 = t.arange(-1100, 1100.01, 2.0, device=dev)
loc = t.arange(-3, 3.01, 0.25, device=dev)
errs, per_trans, blocks = [], [], defaultdict(list)
t0 = time.time()
for gi, g in enumerate(gids):
    sp = by[g][0][3][0].to(dev)                       # (10, 3)
    if A.pos_err > 0:
        gen = t.Generator(device='cpu').manual_seed(int(g))
        sp = sp.clone(); sp[:, :2] += (t.randn(10, 2, generator=gen) * A.pos_err).to(dev)
    c = sp[:, :2].mean(0)
    GX, GY = t.meshgrid(c[0] + g1, c[1] + g1, indexing="ij")
    G = t.stack([GX.ravel(), GY.ravel()], -1)         # (C, 2)
    dG = t.sqrt(((G[:, None, :] - sp[None, :, :2]) ** 2).sum(-1) + sp[None, :, 2] ** 2)
    for X, y, _, _ in by[g]:
        th_true = theta_of[(g, X[0, 0, :5].numpy().tobytes())]
        th = th_true if A.theta == 'known' else float(A.theta)
        R = replica(th, dG)[None]                     # (1, C, S)
        z = tone(X[0, 10:].to(dev))                   # (40, S)
        best = t.cat([misfit(zb[:, None, :], R, th).argmin(1) for zb in z.split(8)])
        p0 = G[best]                                  # (40, 2) coarse
        LX, LY = t.meshgrid(loc, loc, indexing="ij")
        Pc = p0[:, None, :] + t.stack([LX.ravel(), LY.ravel()], -1)[None]       # (40, L, 2)
        dl = t.sqrt(((Pc[:, :, None, :] - sp[None, None, :, :2]) ** 2).sum(-1) + sp[None, None, :, 2] ** 2)
        m = misfit(z[:, None, :], replica(th, dl), th)
        p = Pc[t.arange(40), m.argmin(1)]
        e = (p - y[0, 10:, :2].to(dev)).norm(dim=1).cpu().numpy()
        errs.append(e); per_trans.append((g, e.mean()))
        for k in range(4): blocks[k].append(e[10 * k:10 * k + 10].mean())
    if gi % 25 == 0: print(f"  {gi+1}/{len(gids)} deployments, {time.time()-t0:.0f} s", flush=True)
E_all = np.concatenate(errs)
reg = defaultdict(list)
for g, e in per_trans: reg[data["labels"].get(g, "?")].append(e)
print(f"[{A.tag}] MFP on {len(gids)} B1 deployments, {len(per_trans)} transits: MAE {np.mean([e for _, e in per_trans]):.2f} m, "
      f"median point error {np.median(E_all):.2f} m, 90th pct {np.percentile(E_all, 90):.1f} m, "
      f"share of points within 5 m {np.mean(E_all < 5):.3f}")
print("  by region:", {k: round(float(np.mean(v)), 2) for k, v in reg.items()})
print("  by block of ten targets:", [round(float(np.mean(blocks[k])), 2) for k in range(4)])
