"""Focalization on B2: estimate each held-out deployment's environment from a calibration
transit (its first 10 points, positions known, as the PFN's calibration context), then
build that environment's replica table for MFP (b2_mfp.py --env focal).

Per deployment, the six transits of eval seed 0 (as b2_mfp.py). Two fits: from transit a
(used for every other transit) and from transit b != a (used for transit a itself), so no
transit is localized with an environment fitted on its own points.

Fit: (h0, hr0, c, c2/c, k) in B2's prior box; misfit = the MFP's Cauchy misfit on log
amplitude and wrapped phase over the 10 points x 10 sensors; a random search over the
prior, then Nelder-Mead from the two best candidates.
    python scripts/eval/mfp/b2_focal.py --n 8 --out b2_focal_pilot.npz      # pilot
    python scripts/eval/mfp/b2_focal.py --out b2_focal.npz                  # all 800"""
import os, sys, time, argparse, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORK = os.path.join(REPO, "output", "eval_paper", "mfp", "cache")
sys.path.insert(0, HERE)
LO = np.array([30.0, 1.0, 1470.0, 0.70, 1.4]); HI = np.array([120.0, 5.0, 1540.0, 0.95, 2.0])
SIG, NCTX = 0.05, 10


def tone(X):
    X = X.reshape(-1, 201, 10); k = np.arange(201); w0 = np.pi / 20
    return (np.einsum("pks,k->ps", X, np.cos(w0 * k)) - 1j * np.einsum("pks,k->ps", X, np.sin(w0 * k))) * 2 / 201


def fit(P, S, z, n_rand, n_nm, seed):
    import forward as F
    from scipy.optimize import minimize
    def J(u):
        u = np.asarray(u); pen = float(np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2)) * 1e4
        r = F.replica(F.env_of(LO + np.clip(u, 0, 1) * (HI - LO)), P, S)
        e2 = np.log(np.abs(r) / np.abs(z)) ** 2 + np.angle(r * z.conj()) ** 2
        return float(np.log1p(e2 / SIG ** 2).sum()) + pen
    rng = np.random.default_rng(seed)
    U = rng.random((n_rand, 5)); vals = np.array([J(u) for u in U])
    best = (vals.min(), U[vals.argmin()]); nev = n_rand
    for u0 in U[np.argsort(vals)[:2]]:
        res = minimize(J, u0, method="Nelder-Mead",
                       options={"maxfev": n_nm, "xatol": 1e-3, "fatol": 1e-2, "initial_simplex": None})
        nev += res.nfev
        if res.fun < best[0]:
            best = (res.fun, np.clip(res.x, 0, 1))
    return LO + best[1] * (HI - LO), best[0], nev


def job(a):
    g, trans, n_rand, n_nm = a
    import contextlib, io
    sys.path.insert(0, HERE)
    import b2_tables as TB, forward as F
    rng = np.random.default_rng([g, 77])
    ia, ib = rng.choice(len(trans), 2, replace=False)
    out = {"g": g, "ia": int(ia), "ib": int(ib)}
    t0 = time.time()
    for tag, i in (("A", ia), ("B", ib)):
        X, y, sp = trans[i]
        v, f, nev = fit(y[:NCTX, :2], sp[:, :2], tone(X[:NCTX]), n_rand, n_nm, seed=g * 2 + (tag == "B"))
        F.base.generate_params = F._ORIG_GP
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            h, z = TB.table(F.env_of(v), nproc=1)
        out[f"v{tag}"], out[f"f{tag}"], out[f"n{tag}"], out[f"z{tag}"] = v, f, nev, z
    out["h"], out["sec"] = h, time.time() - t0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--out", default="b2_focal.npz")
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--n-rand", type=int, default=40); ap.add_argument("--n-nm", type=int, default=50)
    A = ap.parse_args()
    import torch as t, pickle, glob
    sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
    import eval_suite as E
    B2 = os.path.join(REPO, "data", "pathD", "hermite_distinct")
    data = E.load_data(f"{B2}/processed/geometry_split", t.device("cpu"), load_cap=18)
    items = E.subsample_test(data["by_gid"], 6, 0)
    by = {}
    for X, y, g, sp in items:
        by.setdefault(g, []).append((X[0].numpy(), y[0].numpy(), sp[0].numpy()))
    gids = sorted(by)[:A.n]
    envs = {}
    for f in sorted(glob.glob(f"{B2}/_manifest*.pkl")):
        envs.update(pickle.load(open(f, "rb"))["environments"])
    res, t0 = [], time.time()
    with Pool(A.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, [(g, by[g], A.n_rand, A.n_nm) for g in gids])):
            e = envs[r["g"]]; tv = np.array([e["h0"], e["hr0"], e["c"], e["c2"] / e["c"], e["k"]])
            r["true"] = tv; res.append(r)
            if i < 10 or i % 50 == 0:
                d = r["vA"] - tv
                print(f"  {i+1}/{len(gids)} g{r['g']}: err h0 {d[0]:+.2f} hr0 {d[1]:+.2f} c {d[2]:+.1f} "
                      f"c2/c {d[3]:+.3f} k {d[4]:+.3f}  misfit {r['fA']:.0f}  evals {r['nA']}  {r['sec']:.0f} s  "
                      f"[{time.time()-t0:.0f} s]", flush=True)
    res.sort(key=lambda r: r["g"])
    os.makedirs(A.work, exist_ok=True)
    np.savez(os.path.join(A.work, A.out), h=res[0]["h"], gids=np.array([r["g"] for r in res]),
             ia=np.array([r["ia"] for r in res]), ib=np.array([r["ib"] for r in res]),
             zA=np.stack([r["zA"] for r in res]), zB=np.stack([r["zB"] for r in res]),
             vA=np.stack([r["vA"] for r in res]), vB=np.stack([r["vB"] for r in res]),
             true=np.stack([r["true"] for r in res]))
    D = np.abs(np.stack([r["vA"] for r in res]) - np.stack([r["true"] for r in res]))
    print("median |error| h0 hr0 c c2/c k:", np.median(D, 0).round(3), " 90th pct:", np.percentile(D, 90, 0).round(3), flush=True)
    print(f"saved {A.out}  total {time.time()-t0:.0f} s", flush=True)
