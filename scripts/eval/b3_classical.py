"""
b3_classical.py
===============

Classical least-squares reference on B3 (calib_10k), from the same delay
features the learned models receive. Only the first-arrival range of each
sensor is used (feature-major layout: X[:, :10] = range_first_km).

    python scripts/eval/b3_classical.py \
        --data data/calib_10k/hermite_distinct/processed_delay/geometry_split/val_data.pkl \
        --out-dir output/eval_paper/b3_classical

Two quantities, six transits per held-out deployment as in eval_suite.py:

  geometry  mean distance between the estimated and the true sensor positions,
            each sensor fitted by linear least squares to the ranges of points
            whose source positions are known. Context protocols:
              calib40 / calib30  N points scattered over ANOTHER transit (B3's
                                 native regime, and the 30-point variant)
              prefix40 / prefix10  the first N points of the SAME transit
  source    mean localization error, every point trilaterated by linearized
            least squares from the ten ranges:
              oracle    true sensor positions
              calib40   sensors from calib40 above
              prefix10  sensors from prefix10, targets = the remaining 40 points
"""
import argparse, os, pickle, csv
from collections import defaultdict
import numpy as np

S_ = 10   # sensors; range_first occupies the first S_ feature columns


def trilaterate(S, r):
    """S (ns, 2) sensor positions, r (npts, ns) ranges -> (npts, 2) sources."""
    A = 2 * (S[0:1] - S[1:])
    b = (r[:, 1:]**2 - r[:, :1]**2) - (np.sum(S[1:]**2, 1) - np.sum(S[0]**2))[None]
    sol, *_ = np.linalg.lstsq(A, b.T, rcond=None)
    return sol.T


def est_sensors(p, r):
    """p (n, 2) known source positions, r (n, ns) ranges -> (ns, 2) sensors.

    |p - s|^2 = r^2 is linear in (s, |s|^2): [-2p, 1] [s; |s|^2] = r^2 - |p|^2.
    """
    A = np.c_[-2 * p, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, r**2 - (p**2).sum(1, keepdims=True), rcond=None)
    return sol[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/calib_10k/hermite_distinct/processed_delay/"
                                      "geometry_split/val_data.pkl")
    ap.add_argument("--out-dir", default="output/eval_paper/b3_classical")
    ap.add_argument("--cap-traj", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by = defaultdict(list)
    for s in pickle.load(open(args.data, "rb")):
        by[int(s["geometry_id"])].append(s)
    rng = np.random.default_rng(args.seed)

    src, geo = defaultdict(list), defaultdict(list)
    for g, items in by.items():
        S = np.asarray(items[0]["sensor_pos"])[:, :2]
        for k in rng.choice(len(items), min(args.cap_traj, len(items)), replace=False):
            it = items[k]
            r = np.asarray(it["X"])[:, :S_] * 1000.0
            y = np.asarray(it["y"])[:, :2]
            src["oracle"].append(np.linalg.norm(trilaterate(S, r) - y, axis=1).mean())
            # calibration transit: any other transit of the same deployment
            j = (k + 1 + rng.integers(len(items) - 1)) % len(items)
            rc = np.asarray(items[j]["X"])[:, :S_] * 1000.0
            yc = np.asarray(items[j]["y"])[:, :2]
            for n in (40, 30):
                sel = np.sort(rng.permutation(len(yc))[:n])
                Se = est_sensors(yc[sel], rc[sel])
                geo[f"calib{n}"].append(np.linalg.norm(Se - S, axis=1).mean())
                if n == 40:
                    src["calib40"].append(np.linalg.norm(trilaterate(Se, r) - y, axis=1).mean())
            for n in (40, 10):
                Se = est_sensors(y[:n], r[:n])
                geo[f"prefix{n}"].append(np.linalg.norm(Se - S, axis=1).mean())
                if n == 10:
                    src["prefix10"].append(
                        np.linalg.norm(trilaterate(Se, r[10:]) - y[10:], axis=1).mean())

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "protocol", "mean", "median", "n"])
        for q, d in (("geometry", geo), ("source", src)):
            for k, v in d.items():
                w.writerow([q, k, f"{np.mean(v):.4f}", f"{np.median(v):.4f}", len(v)])
                print(f"{q:8s} {k:9s} mean {np.mean(v):8.1f} m   median {np.median(v):8.1f} m")
    print(f"{len(by)} deployments, {len(src['oracle'])} transits -> {out}")


if __name__ == "__main__":
    main()
