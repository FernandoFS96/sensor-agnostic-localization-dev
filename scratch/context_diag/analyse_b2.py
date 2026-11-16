"""Per-point comparison of the PFN (calibration context) and the no-context regressor on
the same B2 transits, from ctx_controls.py's npz."""
import glob, json, pickle, sys, numpy as np
tag = sys.argv[1] if len(sys.argv) > 1 else "b2"
Z = np.load(f"output/eval_paper/context_diag/{tag}.npz")
Y, gid = Z["Y"], Z["gid"]
err = lambda k: np.linalg.norm(Z[k] - Y, axis=-1)
er = err("reg__mu")
models = sorted({k.split("__")[0] for k in Z.files if "__calib_n10__" in k})
P = (10, 25, 50, 75, 90, 95, 99)
print("percentiles of the point error [m]    " + "  ".join(f"p{p:<4d}" for p in P) + "   mean   rmse")
def row(n, e): print(f"  {n:34s} " + "  ".join(f"{np.percentile(e, p):5.2f}" for p in P) + f"  {e.mean():5.2f}  {np.sqrt((e**2).mean()):5.2f}")
row("regressor", er)
for m in models:
    for c in ("calib_n10", "calib_n1", "calib_n40"):
        row(f"PFN {m} {c}", err(f"{m}__{c}__mu"))
E = {m: err(f"{m}__calib_n10__mu") for m in models}
if len(models) > 1:
    ens = np.linalg.norm(np.mean([Z[f"{m}__calib_n10__mu"] for m in models], 0) - Y, axis=-1)
    row("PFN seed ensemble", ens)
for m in models:
    row(f"mean of PFN {m} and regressor", np.linalg.norm((Z[f"{m}__calib_n10__mu"] + Z["reg__mu"]) / 2 - Y, axis=-1))
    row(f"oracle min(PFN {m}, regressor)", np.minimum(E[m], er))
print("\nwhere each wins (points binned by the REGRESSOR's error)")
m0 = models[0]; ep = E[m0]
for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)):
    s = (er >= lo) & (er < hi)
    print(f"  regressor error {lo:>4g}-{hi:<6g} share {s.mean():.3f}  regressor {er[s].mean():6.2f}  PFN {ep[s].mean():6.2f}  "
          f"PFN n1 {err(f'{m0}__calib_n1__mu')[s].mean():6.2f}")
print("\nresidual correlation (x and y pooled):")
dr = (Z["reg__mu"] - Y).reshape(-1); 
for m in models:
    dp = (Z[f"{m}__calib_n10__mu"] - Y).reshape(-1)
    print(f"  regressor vs PFN {m}: {np.corrcoef(dr, dp)[0,1]:.3f}")
if len(models) > 1:
    print(f"  PFN {models[0]} vs PFN {models[1]}: {np.corrcoef((Z[f'{models[0]}__calib_n10__mu'] - Y).reshape(-1), (Z[f'{models[1]}__calib_n10__mu'] - Y).reshape(-1))[0,1]:.3f}")
# residual decomposition: deployment / transit / point
def shares(d):
    sq = (d ** 2).sum(); out = []
    ug, inv = np.unique(gid, return_inverse=True)
    dm = np.zeros((len(ug), 2)); cnt = np.bincount(inv)
    for a in range(2): dm[:, a] = np.bincount(inv, d[..., a].mean(1)) / cnt
    return (cnt[:, None] * 50 * dm ** 2).sum() / sq, 50 * (d.mean(1) ** 2).sum() / sq
print("\nshare of squared error that is a per-deployment / per-transit offset:")
print("  regressor      %.3f / %.3f" % shares(Z["reg__mu"] - Y))
for m in models:
    print(f"  PFN {m}         %.3f / %.3f" % shares(Z[f"{m}__calib_n10__mu"] - Y))
# environment
env = {}
for f in glob.glob("data/pathD/hermite_distinct/_manifest_*.pkl"):
    env.update(pickle.load(open(f, "rb"))["environments"])
keys = ("h0", "hr0", "c", "c2", "k")
Ev = np.array([[env[int(g)][k] for k in keys] for g in gid]); Ev[:, 3] /= Ev[:, 2]
rng_src = np.linalg.norm(Y - Z["sp"][:, None, :, :2].mean(2), axis=-1)       # range from the array centroid
print("\nMAE by environment quartile (transit means): regressor | PFN | PFN n1")
e1 = err(f"{m0}__calib_n1__mu")
for j, k in enumerate(("depth h0", "rx height hr0", "sound speed c", "c2/c", "spreading k")):
    q = np.quantile(Ev[:, j], [0, .25, .5, .75, 1]); s = []
    for a in range(4):
        sel = (Ev[:, j] >= q[a]) & (Ev[:, j] <= q[a + 1])
        s.append(f"{er[sel].mean():5.2f}|{ep[sel].mean():5.2f}|{e1[sel].mean():5.2f}")
    print(f"  {k:14s} " + "   ".join(s))
print("\nMAE by source range from the array centroid: regressor | PFN | PFN n1 | share")
for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 1e9)):
    s = (rng_src >= lo) & (rng_src < hi)
    print(f"  {lo:>4g}-{hi:<6g} {er[s].mean():6.2f} | {ep[s].mean():6.2f} | {e1[s].mean():6.2f} | {s.mean():.3f}")
print("\nMAE by block of 10 points: regressor | PFN")
print("  " + "   ".join(f"{er[:, a:a+10].mean():.2f}|{ep[:, a:a+10].mean():.2f}" for a in range(0, 50, 10)))
# does the context gain depend on the distance to the nearest context label?
cl = Y[Z["own"], :10]                                                # (N, 10, 2) context labels
dmin = np.linalg.norm(Y[:, :, None] - cl[:, None], axis=-1).min(-1)
print("\ncontext gain by distance from the target to the nearest context label: PFN n10 | n1 | regressor | share")
for lo, hi in ((0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 1e9)):
    s = (dmin >= lo) & (dmin < hi)
    print(f"  {lo:>4g}-{hi:<6g} {ep[s].mean():6.2f} | {e1[s].mean():6.2f} | {er[s].mean():6.2f} | {s.mean():.3f}")
# label anchoring
for c in ("shift_y10", "shift_y30"):
    d = float(c[7:]); s = (Z[f"{m0}__{c}__mu"] - Z[f"{m0}__calib_n10__mu"])[..., 0] / d
    print(f"\n{c}: prediction shift / label shift: mean {s.mean():.3f}, sd over points {s.std():.3f}, "
          f"p5 {np.percentile(s,5):.2f}, p95 {np.percentile(s,95):.2f}; by range: "
          + ", ".join(f"{s[(rng_src>=lo)&(rng_src<hi)].mean():.2f}" for lo, hi in ((0,50),(50,100),(100,200),(200,1e9))))
