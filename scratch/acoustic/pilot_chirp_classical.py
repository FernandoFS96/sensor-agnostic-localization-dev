"""Classical multilateration on the chirp B2 pilot (pathD_chirp, delay features).

Same transits as eval_calibration.py (val + test pooled, 6 transits per deployment,
targets = points 11-50). Only the first-arrival range of each sensor is used
(X[:, :10] = range_first_km at the nominal c = 1500 m/s). Sensors in a deployment
share one depth, so the linearized trilateration is exact under the slant geometry
(the common vertical offset cancels in the range differences).

  oracle_c    true sensor positions, ranges rescaled with the TRUE sound speed
  nominal     true sensor positions, nominal c = 1500 m/s (no environment knowledge)
  calib10     true sensor positions, range scale fitted on 10 points of ANOTHER transit
              of the deployment with known source positions (the calibration context
              the PFN gets): (a r)^2 = d_h^2 + dz^2, linear in (a^2, dz^2)
"""
import argparse, csv, glob, os, pickle, sys
from collections import defaultdict
import numpy as np

S_ = 10

def trilaterate(S, r):
    A = 2 * (S[0:1] - S[1:])
    b = (r[:, 1:]**2 - r[:, :1]**2) - (np.sum(S[1:]**2, 1) - np.sum(S[0]**2))[None]
    sol, *_ = np.linalg.lstsq(A, b.T, rcond=None)
    return sol.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--raw-root", required=True, help="hermite_distinct dir with the manifests")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap-traj", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = {}
    for f in sorted(glob.glob(os.path.join(a.raw_root, "_manifest*.pkl"))):
        env.update(pickle.load(open(f, "rb")).get("environments", {}))
    by = defaultdict(list)
    for pool in ("val", "test"):
        for s in pickle.load(open(os.path.join(a.data_dir, f"{pool}_data.pkl"), "rb")):
            by[int(s["geometry_id"])].append(s)
    rng = np.random.default_rng(a.seed)
    err = defaultdict(list)
    for g, items in by.items():
        S = np.asarray(items[0]["sensor_pos"])[:, :2]
        c = env[g]["c"] if g in env else None
        for k in rng.choice(len(items), min(a.cap_traj, len(items)), replace=False):
            it = items[k]
            r = np.asarray(it["X"])[:, :S_] * 1000.0
            y = np.asarray(it["y"])[:, :2]
            tgt = slice(10, None)
            e = lambda p: np.linalg.norm(p - y[tgt], axis=1).mean()
            if c is not None:
                err["oracle_c"].append(e(trilaterate(S, r[tgt] * c / 1500.0)))
            err["nominal"].append(e(trilaterate(S, r[tgt])))
            j = (k + 1 + rng.integers(len(items) - 1)) % len(items)
            rc = np.asarray(items[j]["X"])[:, :S_] * 1000.0
            yc = np.asarray(items[j]["y"])[:, :2]
            sel = np.sort(rng.permutation(len(yc))[:10])
            dh2 = (np.linalg.norm(yc[sel, None, :] - S[None], axis=-1)**2).ravel()
            A = np.c_[(rc[sel]**2).ravel(), -np.ones(dh2.size)]
            (a2, _), *_ = np.linalg.lstsq(A, dh2, rcond=None)
            err["calib10"].append(e(trilaterate(S, r[tgt] * np.sqrt(max(a2, 1e-6)))))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "mean", "median", "n"])
        for m, v in err.items():
            w.writerow([m, f"{np.mean(v):.4f}", f"{np.median(v):.4f}", len(v)])
            print(f"{m:9s} MAE pts 11-50: mean {np.mean(v):7.2f} m  median {np.median(v):7.2f} m  (n={len(v)})")
    print(f"{len(by)} deployments, environments known for {sum(g in env for g in by)}")

if __name__ == "__main__":
    main()
