"""
ctx_controls.py
===============

The control protocol applied to the CONTEXT side of a trained PFN, on the
calibration protocol of eval_calibration.py (context = points of another transit
of the same deployment; all 50 points of the target transit scored).

Every condition keeps the target acoustics and changes only the context:

  calib_n<k>     first k points of another transit of the deployment (k=10: default)
  spread10       10 points spread over the donor transit (indices 2, 7, ..., 47)
  no_attn        "empty context": target tokens skip every attention sublayer
                 (out of distribution: the model never trained without a context)
  empty          every context point masked (context_mask all False). The context
                 regressor (ctxreg) then runs as its no-context backbone; a PFN keeps
                 one context token, so this is not an empty context for it
  perm_y         own context, labels permuted among the context points (roll by 5)
  foreign        context (acoustics and labels) from ANOTHER deployment
  foreign_y      own context acoustics, labels of another deployment's transit
  foreign_x      own context labels, acoustics of another deployment's transit
  noise_x        own context labels, acoustics replaced by noise
  shift_y<d>     own context, labels shifted by d meters in x
  self_prefix    first 10 points of the target transit itself (prefix protocol)
  self_spread    10 points spread over the target transit itself (oracle-like;
                 scored on the 40 other points)

The no-context regressor (acoustic_regressor.py, --reg) is scored on the same
transits and points. Saves per-point errors (transits x 50, horizontal error in
meters) for every model and condition in <out>/<tag>.npz and prints a summary.

    python scratch/context_diag/ctx_controls.py --data-dir <split> \
        --models s0=<run>/best.pt s1=<run>/best.pt \
        --reg <regressor.pt> --reg-cache <cache> --tag b2
"""
import argparse, json, os, pickle, sys
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch as t

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval")); sys.path.insert(0, REPO)
import eval_suite as E            # noqa: E402
import acoustic_regressor as R    # noqa: E402

SPREAD = np.arange(2, 50, 5)      # 10 indices over a 50-point transit


def load(data_dir, pools, max_gids):
    """Transits of the given pools in pickle order (the regressor cache's order)."""
    X, Y, SP, gid, th, traj, k0 = [], [], [], [], [], [], []
    for pool in pools:
        S = pickle.load(open(os.path.join(data_dir, f"{pool}_data.pkl"), "rb"))
        keep = None
        if max_gids:
            keep = set(sorted({int(s["geometry_id"]) for s in S})[:max_gids])
        for i, s in enumerate(S):
            g = int(s["geometry_id"])
            if keep is not None and g not in keep:
                continue
            X.append(np.asarray(s["X"], np.float32)); Y.append(np.asarray(s["y"], np.float32))
            SP.append(np.asarray(s["sensor_pos"], np.float32)); gid.append(g)
            th.append(round(float(s["theta"]), 6)); traj.append(int(s["traj_id"])); k0.append((pool, i))
        del S
    return dict(X=t.from_numpy(np.stack(X)), Y=t.from_numpy(np.stack(Y)), SP=t.from_numpy(np.stack(SP)),
                gid=np.array(gid), th=np.array(th), traj=np.array(traj), k0=k0)


def donors(D, seed=0):
    """Own-deployment donor (another trajectory, same theta) and foreign donor
    (a transit of another deployment) per transit."""
    rng = np.random.default_rng([seed, 9999])
    groups = defaultdict(list)
    for i, (g, th) in enumerate(zip(D["gid"], D["th"])):
        groups[(g, th)].append(i)
    N = len(D["gid"])
    own, foreign = np.empty(N, int), np.empty(N, int)
    for i in range(N):
        pool = [j for j in groups[(D["gid"][i], D["th"][i])] if D["traj"][j] != D["traj"][i]]
        own[i] = pool[rng.integers(len(pool))] if pool else -1
        j = i
        while D["gid"][j] == D["gid"][i]:
            j = rng.integers(N)
        foreign[i] = j
    return own, foreign


def context(D, cond, idx, own, foreign, dev):
    """(context_x, context_y in meters, scored-point mask) for transits idx."""
    X, Y = D["X"], D["Y"]
    o, f = own[idx], foreign[idx]
    first = np.arange(10)
    score = np.ones(50, bool)
    if cond.startswith("calib_n"):
        k = int(cond[7:]); cx, cy = X[o, :k], Y[o, :k]
    elif cond == "spread10":
        cx, cy = X[o][:, SPREAD], Y[o][:, SPREAD]
    elif cond in ("no_attn", "empty"):
        cx, cy = X[o, :10], Y[o, :10]
    elif cond == "perm_y":
        cx, cy = X[o, :10], Y[o, :10].roll(5, 1)
    elif cond == "foreign":
        cx, cy = X[f, :10], Y[f, :10]
    elif cond == "foreign_y":
        cx, cy = X[o, :10], Y[f, :10]
    elif cond == "foreign_x":
        cx, cy = X[f, :10], Y[o, :10]
    elif cond == "noise_x":
        cx, cy = X[o, :10], Y[o, :10]
        cx = t.randn_like(cx) * cx.std(dim=(1, 2), keepdim=True)
    elif cond.startswith("shift_y"):
        cx, cy = X[o, :10], Y[o, :10].clone()
        cy[..., 0] += float(cond[7:])
    elif cond == "self_prefix":
        cx, cy = X[idx, :10], Y[idx, :10]
    elif cond == "self_spread":
        cx, cy = X[idx][:, SPREAD], Y[idx][:, SPREAD]
        score[SPREAD] = False
    else:
        raise ValueError(cond)
    return cx.to(dev), cy.to(dev), score


class NoAttn:
    """Context manager: target tokens (positions >= n) skip the attention sublayer
    of every PFN block; context tokens are processed as usual."""
    def __init__(self, model, n):
        self.blocks, self.n = list(model.blocks), n

    def __enter__(self):
        for blk in self.blocks:
            orig = blk.forward

            def fwd(x, key_mask=None, blk=blk, orig=orig):
                y = orig(x, key_mask=key_mask)
                xt = x[:, self.n:]
                return t.cat([y[:, :self.n], xt + blk.ffn(blk.ln2(xt))], 1)
            blk._orig_forward = orig
            blk.forward = fwd

    def __exit__(self, *a):
        for blk in self.blocks:
            blk.forward = blk._orig_forward
            del blk._orig_forward


@t.no_grad()
def run_pfn(b, D, cond, own, foreign, dev, batch=128):
    """Predicted means (N, 50, 2) in meters and the scored-point mask."""
    ym, ys = b["ym"], b["ys"]
    N = len(D["gid"])
    out = np.empty((N, 50, 2), np.float32)
    t.manual_seed(0)
    for a in range(0, N, batch):
        idx = np.arange(a, min(a + batch, N))
        cx, cy, score = context(D, cond, idx, own, foreign, dev)
        cy = (cy - ym) / ys
        tx, sp = D["X"][idx].to(dev), D["SP"][idx].to(dev)
        if cond == "no_attn":
            with NoAttn(b["model"], cx.size(1)):
                mean = b["model"](cx, cy, tx, None, sensor_pos=sp)[0]
        elif cond == "empty":
            mean = b["model"](cx, cy, tx, None, sensor_pos=sp,
                              context_mask=t.zeros(cx.shape[:2], dtype=t.bool, device=dev))[0]
        else:
            mean = b["model"](cx, cy, tx, None, sensor_pos=sp)[0]
        out[idx] = (mean * ys + ym)[..., :2].cpu().numpy()
    return out, score


@t.no_grad()
def run_reg(ckpt, cache, D, pools, feat, loss, dev):
    """The regressor's predictions (N, 50, 2) on the transits of D (cache order = pickle order)."""
    A = SimpleNamespace(encoder="set", pos_code="raw", frame="centroid", aug="rot", d=256, layers=6,
                        loss=loss, var_floor=0.01, feat=feat, tokenizer="mlp", no_ffn=False,
                        summary="readout", pos_err=0.0)
    stats = None
    if feat == "tone":
        z = np.load(os.path.join(cache, "train.npz"))
        la = t.as_tensor(z["tone"]).abs().clamp_min(1e-12).log()
        stats = (float(la.mean()), float(la.std()))
    P = {p: R.load_pool(cache, p, feat, dev, stats, fdev=dev if feat == "tone" else t.device("cpu"))
         for p in pools}
    net = R.Net(A, next(iter(P.values()))["F"].shape[-1], D["SP"].shape[1]).to(dev)
    sd = t.load(ckpt, map_location=dev)
    if not any(k.startswith("body.") for k in sd):       # checkpoints saved before Net wrapped the body
        sd = {"body." + k: v for k, v in sd.items()}
    net.load_state_dict(sd); net.eval()
    out = np.empty((len(D["gid"]), 50, 2), np.float32)
    by_pool = defaultdict(list)
    for i, (pool, k) in enumerate(D["k0"]):
        by_pool[pool].append((i, k))
    for pool, lst in by_pool.items():
        Dp = P[pool]
        ii = np.array([a for a, _ in lst]); kk = t.as_tensor([b for _, b in lst], device=dev)
        assert np.allclose(Dp["y"][kk].cpu().numpy(), D["Y"][ii][..., :2].numpy(), atol=1e-3), "cache order mismatch"
        pts = (kk[:, None] * Dp["ppt"] + t.arange(Dp["ppt"], device=dev)[None]).reshape(-1)
        mus = []
        for a in range(0, len(pts), 8192):
            idx = pts[a:a + 8192]
            pos, F, _, c = R.batch(Dp, idx, A, train=False)
            o = net(pos, F)                            # float32
            mus.append(R.gaussian(o, A)[0] * R.SCALE + c)
        out[ii] = t.cat(mus).view(len(ii), Dp["ppt"], 2).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="label=checkpoint")
    ap.add_argument("--pools", default="val,test")
    ap.add_argument("--max-gids", type=int, default=0, help="first N deployments of each pool (0 = all)")
    ap.add_argument("--conds", default="calib_n10,calib_n1,calib_n2,calib_n5,calib_n20,calib_n40,spread10,"
                    "no_attn,perm_y,foreign,foreign_y,foreign_x,noise_x,shift_y10,shift_y30,"
                    "self_prefix,self_spread")
    ap.add_argument("--reg", default=None); ap.add_argument("--reg-cache", default=None)
    ap.add_argument("--reg-feat", default="tone"); ap.add_argument("--reg-loss", default="nll")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "output/eval_paper/context_diag"))
    A = ap.parse_args()
    dev = t.device("cuda")
    pools = A.pools.split(",")
    D = load(A.data_dir, pools, A.max_gids)
    own, foreign = donors(D)
    Y = D["Y"][..., :2].numpy()
    print(f"[{A.tag}] {len(D['gid'])} transits, {len(set(D['gid']))} deployments, pools {pools}", flush=True)
    save = dict(gid=D["gid"], traj=D["traj"], th=D["th"], Y=Y, own=own, foreign=foreign,
                sp=D["SP"].numpy())
    conds = A.conds.split(",")
    if A.reg:
        mu = run_reg(A.reg, A.reg_cache, D, pools, A.reg_feat, A.reg_loss, dev)
        save["reg__mu"] = mu
        e = np.linalg.norm(mu - Y, axis=-1)
        print(f"  {'regressor':10s} {'-':12s} all50 {e.mean():7.2f}  late40 {e[:, 10:].mean():7.2f}  "
              f"median pt {np.median(e):6.2f}", flush=True)
    for spec in A.models:
        lbl, path = spec.split("=", 1)
        b = E.load_model(path, dev)
        base = None
        for c in conds:
            mu, score = run_pfn(b, D, c, own, foreign, dev)
            save[f"{lbl}__{c}__mu"] = mu
            e = np.linalg.norm(mu - Y, axis=-1)
            late = score.copy(); late[:10] = False
            msg = (f"  {lbl:10s} {c:12s} all50 {e[:, score].mean():7.2f}  late40 {e[:, late].mean():7.2f}  "
                   f"median pt {np.median(e[:, score]):6.2f}")
            if c == "calib_n10":
                base = mu
            if c.startswith("shift_y") and base is not None:
                d = float(c[7:]); s = (mu - base).mean((0, 1))
                msg += f"  mean prediction shift ({s[0]:.2f}, {s[1]:.2f}) m = {s[0] / d:.3f} of the label shift"
            print(msg, flush=True)
        del b
    os.makedirs(A.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(A.out_dir, f"{A.tag}.npz"), **save)
    print(f"[{A.tag}] wrote {A.out_dir}/{A.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
