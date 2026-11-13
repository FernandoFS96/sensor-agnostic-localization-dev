"""
eval_calibration.py
===================

Evaluation with CALIBRATION context: the context is the first N points of ANOTHER
transit of the same deployment (another trajectory under the same channel option
theta), as in calibration-regime training; the targets are all points of the
transit being localized. The context then reveals nothing about the target's
path, so the trajectory-continuation shortcut of the prefix protocol is gone.

    python scripts/eval/eval_calibration.py \
        --data-dir data/pfn10kd/hermite_distinct/processed/geometry_split \
        --models pfn=<run>/best.pt pfn=<run>/best.pt ... \
        --out-dir output/eval_paper/b1_calibctx

Split models (PFN, attentive NPs) take the calibration points as their context.
Recurrent (indexed) models read one sequence: the calibration points, then the
target transit; context indices are the first N steps, targets the rest.
Transits per deployment and the held-out pool follow eval_suite.py; each eval
seed redraws the donor transit and the ablation noise.

Metrics per model: MAE over all targets (overall + region labels), MAE over
targets 11-50 (the prefix protocol's targets), MAE by block of ten targets, the
noise ratio (target acoustics replaced by noise / intact) and the coverage of the
50/90/95% predictive ellipses on x/y (fraction of targets whose squared
Mahalanobis distance under the predicted Gaussian is below the chi2(2) quantile;
a calibrated model scores 0.50/0.90/0.95). Aggregates: mean
over eval seeds per training seed, then mean and sample sd over training seeds.
"""
import argparse, os, sys, csv, json, pickle
from collections import defaultdict
import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E


def load_pool(data_dir, cap_load):
    """Held-out samples grouped by geometry (val + test), with theta and traj id."""
    by_gid = defaultdict(list)
    for pool in ("val", "test"):
        p = os.path.join(data_dir, f"{pool}_data.pkl")
        if not os.path.exists(p):
            continue
        for s in pickle.load(open(p, "rb")):
            g = int(s["geometry_id"])
            if len(by_gid[g]) >= cap_load:
                continue
            by_gid[g].append({"X": np.asarray(s["X"], np.float32), "y": np.asarray(s["y"], np.float32),
                              "sp": np.asarray(s["sensor_pos"], np.float32),
                              "th": round(float(s["theta"]), 6), "traj": int(s["traj_id"])})
    return by_gid


def pairs(by_gid, cap_traj, eseed, self_donor=False, foreign_donor=False):
    """(target, donor) per selected transit; donor = another trajectory, same theta.
    self_donor=True makes the target its own donor: the prefix protocol, used as a
    check that this harness reproduces eval_suite for the split models.
    foreign_donor=True takes the donor from ANOTHER deployment: the wrong-context
    control (a model that uses its context should get worse)."""
    rng = np.random.default_rng([eseed, 9999])
    out = []
    gids = list(by_gid)
    for g, items in by_gid.items():
        idx = rng.permutation(len(items))[:min(cap_traj, len(items))]
        for i in idx:
            tg = items[i]
            if self_donor:
                out.append((g, tg, tg)); continue
            if foreign_donor:
                og = g
                while og == g:
                    og = gids[rng.integers(len(gids))]
                out.append((g, tg, by_gid[og][rng.integers(len(by_gid[og]))])); continue
            pool = [k for k, it in enumerate(items) if it["th"] == tg["th"] and it["traj"] != tg["traj"]]
            if pool:
                out.append((g, tg, items[pool[rng.integers(len(pool))]]))
    return out


@t.no_grad()
def predict(b, Xt, Xd, yd, sp, nctx, noise=False, mask=None):
    """Xt (B, P, F) target transits, Xd/yd donor transits -> (B, P, D) means and
    variances, in physical units."""
    ym, ys = b["ym"], b["ys"]
    cy = yd[:, :nctx]
    cy = (cy - ym) / ys if ym is not None else cy
    tx = Xt
    if noise:
        tx = t.randn_like(Xt) * Xt.std(dim=(1, 2), keepdim=True)
    spk = sp if b["spatial"] else None
    kw = {"sensor_mask": mask} if (mask is not None and b["spatial"]) else {}
    if b["conv"] == "split":
        mean, var = b["model"](Xd[:, :nctx], cy, tx, None, sensor_pos=sp, **kw)[:2]
    else:
        seq = t.cat([Xd[:, :nctx], tx], 1)
        P = Xt.size(1)
        ci = t.arange(nctx, device=Xt.device); ti = t.arange(nctx, nctx + P, device=Xt.device)
        mean, var = b["model"](seq, ci, cy, ti, None, sensor_pos=spk, **kw)[:2]
    return (mean * ys + ym, var * ys ** 2) if ym is not None else (mean, var)


# chi2 quantiles with 2 degrees of freedom: -2 ln(1 - p)
COVERAGE = {p: -2.0 * np.log(1.0 - p) for p in (0.5, 0.9, 0.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-ctx", type=int, default=10)
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--cap-traj", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--self-donor", action="store_true", help="prefix-protocol check")
    ap.add_argument("--foreign-donor", action="store_true",
                    help="wrong-context control: the calibration transit comes from another deployment")
    ap.add_argument("--pos-err", type=float, default=0.0,
                    help="sd [m] of a horizontal error added to the sensor positions the model is given")
    ap.add_argument("--drop", type=int, default=0, help="sensors removed per transit (random, seeded)")
    args = ap.parse_args()
    dev = t.device("cuda" if t.cuda.is_available() else "cpu")

    models = defaultdict(list)
    for spec in args.models:
        lbl, path = spec.split("=", 1)
        models[lbl].append(E.load_model(path, dev))
    print("[calib] model seeds:", {k: len(v) for k, v in models.items()}, flush=True)
    by_gid = load_pool(args.data_dir, max(15, args.cap_traj * 3))
    labels = {int(k): v.get("region", "?") for k, v in
              json.load(open(os.path.join(args.data_dir, "splits.json")))["labels"].items()}

    rows = []      # (model, metric, eval_seed, train_seed, value)
    for es in (int(z) for z in args.eval_seeds.split(",")):
        pr = pairs(by_gid, args.cap_traj, es, args.self_donor, args.foreign_donor)
        print(f"[calib] eval seed {es}: {len(pr)} transits", flush=True)
        for lbl, bs in models.items():
            for mi, b in enumerate(bs):
                t.manual_seed(es)
                per, blocks, late, noisy, r2 = [], defaultdict(list), [], [], []
                for k in range(0, len(pr), args.batch):
                    chunk = pr[k:k + args.batch]
                    Xt = t.tensor(np.stack([c[1]["X"] for c in chunk]), device=dev)
                    yt = t.tensor(np.stack([c[1]["y"] for c in chunk]), device=dev)
                    Xd = t.tensor(np.stack([c[2]["X"] for c in chunk]), device=dev)
                    yd = t.tensor(np.stack([c[2]["y"] for c in chunk]), device=dev)
                    sp = t.tensor(np.stack([c[1]["sp"] for c in chunk]), device=dev)
                    gen = t.Generator().manual_seed(1000 * es + k)
                    if args.pos_err > 0:
                        sp = sp.clone()
                        sp[..., :2] += (t.randn(sp.shape[:-1] + (2,), generator=gen) * args.pos_err).to(dev)
                    mask = None
                    if args.drop > 0:
                        S = sp.shape[1]
                        keep = t.rand(len(chunk), S, generator=gen).argsort(1) >= args.drop
                        mask = keep.to(dev)
                    mu, var = predict(b, Xt, Xd, yd, sp, args.n_ctx, mask=mask)
                    e = (mu - yt).norm(dim=-1)                                                       # (B, P)
                    en = (predict(b, Xt, Xd, yd, sp, args.n_ctx, noise=True, mask=mask)[0] - yt).norm(dim=-1)
                    r2.append((((mu - yt)[..., :2] ** 2) / var[..., :2]).sum(-1).flatten().cpu())
                    per += [(c[0], v) for c, v in zip(chunk, e.mean(1).tolist())]
                    late += e[:, 10:].mean(1).tolist(); noisy += en.mean(1).tolist()
                    for j in range(0, e.size(1), 10):
                        blocks[j // 10] += e[:, j:j + 10].mean(1).tolist()
                allm = float(np.mean([v for _, v in per]))
                rows += [(lbl, "overall", es, mi, allm), (lbl, "late40", es, mi, float(np.mean(late))),
                         (lbl, "noise_ratio", es, mi, float(np.mean(noisy)) / allm)]
                r2 = t.cat(r2).numpy()
                rows += [(lbl, f"cov{round(100 * p)}", es, mi, float(np.mean(r2 <= q)))
                         for p, q in COVERAGE.items()]
                buck = defaultdict(list)
                for g, v in per:
                    buck[labels.get(g, "?")].append(v)
                rows += [(lbl, r, es, mi, float(np.mean(v))) for r, v in buck.items() if r != "?"]
                rows += [(lbl, f"block_{k}", es, mi, float(np.mean(v))) for k, v in blocks.items()]
                print(f"  {lbl:8s} seed {mi}: MAE {allm:7.2f}  late40 {np.mean(late):7.2f}  "
                      f"noise x{np.mean(noisy) / allm:5.2f}  coverage 50/90/95 "
                      + "/".join(f"{np.mean(r2 <= q):.3f}" for q in COVERAGE.values()), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results_per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "metric", "eval_seed", "train_seed", "value"])
        w.writerows([(a, m, es, mi, f"{v:.4f}") for a, m, es, mi, v in rows])
    acc = defaultdict(lambda: defaultdict(list))
    for a, m, es, mi, v in rows:
        acc[(a, m)][mi].append(v)
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "metric", "mean", "sd_train_seeds", "n_train_seeds"])
        for (a, m), per in acc.items():
            v = [float(np.mean(z)) for z in per.values()]
            w.writerow([a, m, f"{np.mean(v):.4f}", f"{np.std(v, ddof=1) if len(v) > 1 else 0.0:.4f}", len(v)])
    print(f"[calib] wrote {args.out_dir}/results.csv", flush=True)


if __name__ == "__main__":
    main()
