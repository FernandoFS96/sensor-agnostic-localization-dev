"""Matched-field processing on B2 (a new environment per deployment), all 800 held-out
deployments, six transits each, every point of the transit (no context is used); prints
the MAE over all points and over points 11-50 (the paper protocol), with region splits.
  --env nominal   one replica table for the centre of the environment prior (b2_nominal.npz)
  --env true      each deployment's own environment (an oracle: it is what B2 hides; b2_true.npz)
  --env focal     the environment fitted to a calibration transit (b2_focal.py); the
                  calibration transit itself uses the fit from a second transit
  --proc cplx | level_demean   (level + phase | level only, no source level)
Replicas are tabulated against horizontal source-sensor distance."""
import os, sys, argparse, time, numpy as np, torch as t
from collections import defaultdict
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(REPO, "output", "eval_paper", "mfp", "cache")
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
import eval_suite as E

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="nominal"); ap.add_argument("--proc", default="cplx")
ap.add_argument("--n", type=int, default=10000); ap.add_argument("--tag", default="")
ap.add_argument("--data-dir", default=os.path.join(REPO, "data/pathD/hermite_distinct/processed/geometry_split"))
ap.add_argument("--work", default=WORK)
ap.add_argument("--focal", default="b2_focal.npz")
A = ap.parse_args()
dev = t.device("cuda")
HG = np.arange(0.0, 1300.0, 0.05)
def grid(h, z):
    return t.tensor(np.interp(HG, h, z.real) + 1j * np.interp(HG, h, z.imag), dtype=t.complex64, device=dev)
if A.env == "nominal":
    T = np.load(os.path.join(A.work, "b2_nominal.npz")); NOM = grid(T["h"], T["z"]); tab_of = lambda g: NOM
elif A.env == "focal":
    T = np.load(os.path.join(A.work, A.focal)); idx = {int(g): i for i, g in enumerate(T["gids"])}
    tab_of = lambda g, k=0: grid(T["h"], (T["zB"] if k == int(T["ia"][idx[g]]) else T["zA"])[idx[g]])
else:
    T = np.load(os.path.join(A.work, "b2_true.npz")); idx = {int(g): i for i, g in enumerate(T["gids"])}
    tab_of = lambda g: grid(T["h"], T["z"][idx[g]])
SIG = 0.05
w0 = np.pi / 20; nn_ = np.arange(201)
CS = t.tensor(np.stack([np.cos(w0 * nn_), np.sin(w0 * nn_)]), dtype=t.float32, device=dev)

def tone(X):
    X = X.view(-1, 201, 10)
    return t.complex(t.einsum("pns,n->ps", X, CS[0]), -t.einsum("pns,n->ps", X, CS[1])) * 2 / 201

def replica(tab, h):
    i = (h / 0.05).clamp(0, len(HG) - 2); i0 = i.floor().long(); f = (i - i0).to(t.float32)
    return tab[i0] * (1 - f) + tab[i0 + 1] * f

def misfit(z, r):
    e_a = t.log(r.abs() + 1e-12) - t.log(z.abs() + 1e-12)
    if A.proc == "level_demean":
        e_a = e_a - e_a.mean(-1, keepdim=True)
    e2 = e_a ** 2 if A.proc != "cplx" else e_a ** 2 + t.angle(r * z.conj()) ** 2
    return t.log1p(e2 / SIG ** 2).sum(-1)

data = E.load_data(A.data_dir, t.device("cpu"), load_cap=18)
items = E.subsample_test(data["by_gid"], 6, 0)
by = defaultdict(list)
for it in items: by[it[2]].append(it)
gids = sorted(by)[:A.n]
if A.env == "focal":
    gids = [g for g in gids if g in idx]
g1 = t.arange(-1100, 1100.01, 2.0, device=dev); loc = t.arange(-3, 3.01, 0.25, device=dev)
LX, LY = t.meshgrid(loc, loc, indexing="ij"); LOC = t.stack([LX.ravel(), LY.ravel()], -1)
per, allp, t0 = [], [], time.time()
for gi, g in enumerate(gids):
    tab = tab_of(g)
    sp = by[g][0][3][0].to(dev); c = sp[:, :2].mean(0)
    GX, GY = t.meshgrid(c[0] + g1, c[1] + g1, indexing="ij"); G = t.stack([GX.ravel(), GY.ravel()], -1)
    R = replica(tab, (G[:, None, :] - sp[None, :, :2]).norm(dim=-1))[None]
    for k, (X, y, _, _) in enumerate(by[g]):
        if A.env == "focal":
            tab = tab_of(g, k)
            R = replica(tab, (G[:, None, :] - sp[None, :, :2]).norm(dim=-1))[None]
        z = tone(X[0].to(dev))                                         # all 50 points
        p0 = G[t.cat([misfit(zb[:, None, :], R).argmin(1) for zb in z.split(8)])]
        Pc = p0[:, None, :] + LOC[None]
        m = misfit(z[:, None, :], replica(tab, (Pc[:, :, None, :] - sp[None, None, :, :2]).norm(dim=-1)))
        e = (Pc[t.arange(len(z)), m.argmin(1)] - y[0, :, :2].to(dev)).norm(dim=1).cpu().numpy()
        per.append((g, e.mean(), e[10:].mean())); allp.append(e)
    if gi % 100 == 0: print(f"  {gi+1}/{len(gids)} deployments, {time.time()-t0:.0f} s", flush=True)
allp = np.concatenate(allp); reg = defaultdict(list); reg40 = defaultdict(list)
for g, e, e40 in per:
    reg[data["labels"].get(g, "?")].append(e); reg40[data["labels"].get(g, "?")].append(e40)
print(f"[{A.tag}] MFP on {len(gids)} B2 deployments, {len(per)} transits: MAE all points {np.mean([p[1] for p in per]):.2f} m, "
      f"points 11-50 {np.mean([p[2] for p in per]):.2f} m, median point {np.median(allp):.2f} m, "
      f"within 5 m {np.mean(allp < 5):.3f}; by region {dict((k, round(float(np.mean(v)), 2)) for k, v in reg.items())}", flush=True)
print("  by region, points 11-50:", {k: round(float(np.mean(v)), 2) for k, v in reg40.items()}, flush=True)
