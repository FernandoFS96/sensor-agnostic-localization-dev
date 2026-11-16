"""
acoustic_regressor.py
=====================

How much of the B1 acoustics a plain network can use. A supervised regressor maps
ONE point's received records at the 10 sensors, plus the sensor positions, to the
source position: no context and no trajectory, the same inputs as the MFP baseline.
B1's environment is fixed, so this is a fixed mapping; the MFP with the true
environment reaches about 4 m on it and the NPs about 20 m.

The model is a transformer over the sensors (one token per sensor: its position
relative to the array centroid plus its record features, and one readout token),
trained on every point of the training split with random rotations and reflections
about the array centroid (the channel depends only on source-sensor geometry).
Two front ends:
  --feat tone   log amplitude and phase of the demodulated tone (the MFP's input)
  --feat raw    the 201-sample record, as the NPs receive it

Scored on every held-out transit (val + test pools, which is what eval_suite's
subsample_test returns for 6 transits per deployment, in B1 and B2), points 11-50
as the B1 MFP and all 50 points as the B2 MFP.

Ablations toward the NPs' front end (SpatialEncoder in src/models/anp.py):
  --pos-code fourier  sensor positions as the NPs' Fourier code (8 bands, 10-1000 m)
                      instead of plain coordinates
  --frame absolute    positions and targets in the data frame (minus the training
                      mean, a constant shift), not relative to the array centroid
  --aug trans         the NPs' translation augmentation, U(-150, 150) m in x and y,
                      instead of rotations and reflections
  --encoder np        the NPs' SpatialEncoder itself (paper settings: raw records,
                      standardized, 3 attention layers, attention pooling, width 128)
                      with an MLP head, in place of the sensor transformer
and inside the sensor transformer, one SpatialEncoder design choice at a time:
  --d 128 / --layers 3  its width and depth (SpatialEncoder: 128, 3)
  --tokenizer linear    one linear map per sensor token, not a two-layer MLP
  --no-ffn              attention-only mixing layers, no feed-forward sublayer
  --summary pool        one attention pool after the layers, not a readout token

    python scripts/eval/acoustic_regressor.py extract --cache <dir> [--data-dir <split>]
    python scripts/eval/acoustic_regressor.py train --cache <dir> --feat tone \
        --out-dir output/eval_paper/baselines/b1
"""
import argparse, csv, json, os, pickle, sys, time
import numpy as np
import torch as t
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from src.models.anp import FourierPositionEncoding, SpatialEncoder  # noqa: E402

DATA = "data/pfn10kd/hermite_distinct/processed/geometry_split"
SCALE = 100.0          # meters per network unit
W0 = np.pi / 20        # tone frequency of the sinusoid records [rad/sample]


# --------------------------------------------------------------------------- #
# Cache: the pickles hold float32 records; keep tone, labels and a float16 copy
# --------------------------------------------------------------------------- #
def extract(data_dir, cache):
    os.makedirs(cache, exist_ok=True)
    xscale = None
    for pool in ("val", "test", "train"):
        t0 = time.time()
        S = pickle.load(open(os.path.join(data_dir, f"{pool}_data.pkl"), "rb"))
        N = len(S); ppt, F = np.asarray(S[0]["X"]).shape
        ns = np.asarray(S[0]["sensor_pos"]).shape[0]; tau = F // ns
        if xscale is None:     # records are ~6e-3; scale to unit std before float16
            xscale = float(np.stack([np.asarray(s["X"]) for s in S[:300]]).std())
        k = np.arange(tau)
        basis = ((np.cos(W0 * k) - 1j * np.sin(W0 * k)) * 2 / tau).astype(np.complex64)
        raw = np.lib.format.open_memmap(os.path.join(cache, f"{pool}_raw.npy"), "w+", np.float16, (N, ppt, F))
        tone = np.empty((N, ppt, ns), np.complex64); Y = np.empty((N, ppt, 2), np.float32)
        SP = np.empty((N, ns, 2), np.float32); gid = np.empty(N, np.int64); th = np.empty(N, np.float32)
        for i, s in enumerate(S):
            X = np.asarray(s["X"], np.float32)
            raw[i] = X / xscale
            tone[i] = np.einsum("pks,k->ps", X.reshape(ppt, tau, ns), basis)   # feat is tau-major
            Y[i] = np.asarray(s["y"], np.float32)[:, :2]
            SP[i] = np.asarray(s["sensor_pos"], np.float32)[:, :2]
            gid[i] = int(s["geometry_id"]); th[i] = float(s["theta"])
        raw.flush(); del raw, S
        np.savez(os.path.join(cache, f"{pool}.npz"), tone=tone, y=Y, sp=SP, gid=gid, theta=th,
                 xscale=xscale, tau=tau)
        print(f"{pool}: {N} transits x {ppt} points, {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def f32(head, h):
    """An output head in float32. Training runs under a bfloat16 autocast, whose
    8 significant bits would round a coordinate of 200 to 400 m to steps of 1.5 m."""
    with t.autocast("cuda", enabled=False):
        return head(h.float())


class AttnOnlyLayer(nn.Module):
    """Pre-norm self-attention with a residual and no feed-forward sublayer, as the
    mixing layers of the NPs' SpatialEncoder (--no-ffn)."""
    def __init__(self, d, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, h, kpm=None):
        x = self.norm(h)
        return h + self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)[0]


class SensorSetRegressor(nn.Module):
    """Ablation flags toward the NPs' SpatialEncoder, one design choice each:
    tok='linear' (one linear map per sensor token instead of a two-layer MLP),
    ffn=False (attention-only mixing layers), summary='pool' (one attention pool with
    a learned query after the mixing layers instead of a readout token that takes part
    in every layer). Width and depth are d and layers."""
    def __init__(self, fin, d=256, layers=6, heads=8, pin=2, out_dim=2, tok="mlp", ffn=True,
                 summary="readout"):
        super().__init__()
        self.summary = summary
        self.inp = (nn.Sequential(nn.Linear(fin + pin, d), nn.GELU(), nn.Linear(d, d)) if tok == "mlp"
                    else nn.Linear(fin + pin, d))
        if summary == "readout":
            self.cls = nn.Parameter(t.zeros(1, 1, d))
        else:
            self.query = nn.Parameter(t.randn(1, 1, d) * 0.02)
            self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        if ffn:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.enc = nn.ModuleList([AttnOnlyLayer(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))

    def mix(self, h, kpm):
        if isinstance(self.enc, nn.TransformerEncoder):
            return self.enc(h, src_key_padding_mask=kpm)
        for layer in self.enc:
            h = layer(h, kpm)
        return h

    def forward(self, pos, feat, mask=None):           # (B, S, 2), (B, S, fin), (B, S) True = present
        h = self.inp(t.cat([pos, feat], -1))
        kpm = None if mask is None else ~mask          # PyTorch convention: True = ignore
        if self.summary == "pool":
            h = self.mix(h, kpm)
            q = self.query.expand(len(h), -1, -1)
            return f32(self.out, self.pool(q, h, h, key_padding_mask=kpm, need_weights=False)[0][:, 0])
        h = t.cat([self.cls.expand(len(h), -1, -1), h], 1)
        if kpm is not None:
            kpm = t.cat([t.zeros(len(h), 1, dtype=t.bool, device=h.device), kpm], 1)
        return f32(self.out, self.mix(h, kpm)[:, 0])


class Net(nn.Module):
    """Position code + body. Takes sensor positions in meters (in the chosen frame).
    With --loss nll the body also returns two raw variances (diagonal Gaussian)."""
    def __init__(self, A, fin, ns):
        super().__init__()
        self.kind, self.ns = A.encoder, ns
        self.pe = FourierPositionEncoding(8, 10.0, 1000.0, in_dim=2) if A.pos_code == "fourier" else None
        if A.encoder == "set":
            pin = self.pe.out_dim if self.pe is not None else 2
            self.body = SensorSetRegressor(fin, A.d, A.layers, pin=pin,
                                           out_dim=4 if A.loss == "nll" else 2, tok=A.tokenizer,
                                           ffn=not A.no_ffn, summary=A.summary)
        else:                                          # the NPs' front end, paper settings
            self.enc = SpatialEncoder(fin * ns, ns, 128, tokenize=True, use_position=True,
                                      use_attention=True, n_attn_layers=3, pooling="attention",
                                      n_fourier_bands=8, min_wavelength=10.0, max_wavelength=1000.0,
                                      pos_dim=2, dropout=0.1, norm_acoustic="standardize")
            self.enc.set_acoustic_stats(0.0, 1.0)     # the raw cache is already unit-scaled
            self.head = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256),
                                      nn.GELU(), nn.Linear(256, 2))

    def forward(self, pos, feat, mask=None):           # (B, S, 2) meters, (B, S, fin)
        if self.kind == "set":
            p = self.pe(pos) if self.pe is not None else pos / SCALE
            return self.body(p, feat, mask)
        x = feat.transpose(1, 2).reshape(len(feat), 1, -1)   # back to tau-major flat
        return f32(self.head, self.enc(x, pos, mask)[:, 0])


def gaussian(out, A):
    """Network output -> (mean, variance) in network units; variance None for --loss euclid.
    The floor is --var-floor m^2, as model.var_floor_phys in the NP trainer."""
    if A.loss != "nll":
        return out, None
    return out[:, :2], A.var_floor / SCALE ** 2 + nn.functional.softplus(out[:, 2:])


def drop_mask(B, S, p, min_keep, dev):
    """(B, S) bool, True = present: each sensor dropped with probability p, at most
    S - min_keep of them (the lowest draws), as data.sensor_drop.mode=bernoulli."""
    u = t.rand(B, S, device=dev)
    return ~((u < p) & (u.argsort(1).argsort(1) < S - min_keep))


def load_pool(cache, pool, feat, dev, stats=None, fdev=None):
    """Per-point features (N*ppt, S, fin) on fdev (default dev), plus labels and array
    geometry on dev. The raw held-out pools stay on the CPU to leave GPU room."""
    fdev = dev if fdev is None else fdev
    z = np.load(os.path.join(cache, f"{pool}.npz"))
    N, ppt, ns = z["tone"].shape
    if feat == "tone":
        tone = t.as_tensor(z["tone"])
        la = tone.abs().clamp_min(1e-12).log()
        if stats is None:
            stats = (float(la.mean()), float(la.std()))
        ph = tone.angle()
        F = t.stack([(la - stats[0]) / stats[1], ph.cos(), ph.sin()], -1)       # (N, ppt, S, 3)
        F = F.reshape(N * ppt, ns, 3).to(fdev)
    else:
        tau = int(z["tau"])
        R = np.load(os.path.join(cache, f"{pool}_raw.npy"), mmap_mode="r")
        F = t.empty((N * ppt, ns, tau), dtype=t.float16, device=fdev)
        for a in range(0, N, 2000):                    # (n, ppt, tau*S) tau-major -> (n*ppt, S, tau)
            b = t.as_tensor(np.ascontiguousarray(R[a:a + 2000]).copy()).view(-1, ppt, tau, ns)
            F[a * ppt:(a + len(b)) * ppt] = b.transpose(2, 3).reshape(-1, ns, tau).to(fdev)
    return dict(F=F, y=t.as_tensor(z["y"]).to(dev), sp=t.as_tensor(z["sp"]).to(dev),
                gid=z["gid"], N=N, ppt=ppt, stats=stats)


def batch(D, idx, A, train=True):
    """Points idx -> (sensor pos [m], features, target [network units], origin [m]).
    The origin is the array centroid (--frame centroid) or the training mean (absolute)."""
    n, B, dev = idx // D["ppt"], len(idx), idx.device
    sp = D["sp"][n]
    if train and A.pos_err > 0:                        # mis-surveyed array: sd ~ U(0, pos_err) per point
        sp = sp + t.randn_like(sp) * t.rand(B, 1, 1, device=dev) * A.pos_err
    o = sp.mean(1) if A.frame == "centroid" else D["origin"].expand(B, 2)
    if train and A.aug == "trans":                     # shift the scene == shift the origin back
        o = o - (t.rand(B, 2, device=dev) * 2 - 1) * A.trans_m
    pos = sp - o[:, None]
    tgt = (D["y"].view(-1, 2)[idx] - o) / SCALE
    if train and A.aug == "rot":
        a = t.rand(B, device=dev) * 2 * np.pi
        R = t.stack([t.stack([a.cos(), -a.sin()], -1), t.stack([a.sin(), a.cos()], -1)], -2)
        R[:, :, 0] *= t.where(t.rand(B, 1, device=dev) < 0.5, -1.0, 1.0)
        pos = pos @ R.transpose(1, 2); tgt = (tgt[:, None] @ R.transpose(1, 2))[:, 0]
    F = D["F"][idx.to(D["F"].device)].to(dev).float()
    return pos, F, tgt, o


# chi2 quantiles with 2 degrees of freedom: -2 ln(1 - p)
COVERAGE = {p: -2.0 * np.log(1.0 - p) for p in (0.5, 0.9, 0.95)}


@t.no_grad()
def predict(net, D, A, n_transits=None):
    """Point errors (n, ppt) in meters for the first n_transits transits, and the squared
    Mahalanobis distances (n, ppt) under the predicted Gaussian (None for --loss euclid).
    D["mask"] (transits, S), if present, removes sensors per transit."""
    n = D["N"] if n_transits is None else min(n_transits, D["N"])
    errs, r2 = [], []
    for a in range(0, n * D["ppt"], 8192):
        idx = t.arange(a, min(a + 8192, n * D["ppt"]), device=D["sp"].device)
        pos, F, _, c = batch(D, idx, A, train=False)
        mask = D["mask"][idx // D["ppt"]] if "mask" in D else None
        out = net(pos, F, mask)                        # float32: bfloat16 is for training only
        mu, var = gaussian(out, A)
        d = mu * SCALE + c - D["y"].view(-1, 2)[idx]
        errs.append(d.norm(dim=-1))
        if var is not None:
            r2.append((d ** 2 / (var * SCALE ** 2)).sum(-1))
    E = t.cat(errs).view(n, D["ppt"]).cpu().numpy()
    return E, (t.cat(r2).view(n, D["ppt"]).cpu().numpy() if r2 else None)


def robustness(net, ho, A, out_csv):
    """Every held-out transit, all 50 points (the B2 convention), intact and under the
    perturbations of eval_calibration: horizontal sensor-position error (sd per axis,
    one draw per sensor and transit) and k of the sensors removed per transit."""
    labels = {int(k): v.get("region", "?")
              for k, v in json.load(open(os.path.join(A.data_dir, "splits.json")))["labels"].items()}
    reg = np.array([labels.get(int(g), "?") for D in ho for g in D["gid"]])
    cases = [("intact", 0.0, 0)] + [(f"pos_err_{e:g}m", e, 0) for e in (2.0, 5.0)] \
        + [(f"drop_{k}", 0.0, k) for k in (3, 5, 7)]
    rows = []
    for name, pe, k in cases:
        g = t.Generator().manual_seed(1234)
        E, R2 = [], []
        for D in ho:
            Dk = dict(D)
            if pe > 0:
                Dk["sp"] = D["sp"] + (t.randn(D["sp"].shape, generator=g) * pe).to(D["sp"].device)
            if k > 0:
                S = D["sp"].shape[1]
                Dk["mask"] = (t.rand(D["N"], S, generator=g).argsort(1) >= k).to(D["sp"].device)
            e, r2 = predict(net, Dk, A)
            E.append(e); R2.append(r2)
        E = np.concatenate(E)
        row = {"case": name, "mae_all50_m": E.mean()}
        for r in ("deep", "shallow", "interp", "extrap"):
            if (reg == r).any():
                row[r] = E[reg == r].mean()
        if R2[0] is not None:
            R2 = np.concatenate(R2)
            row.update({f"cov{round(100 * p)}": float(np.mean(R2 <= q)) for p, q in COVERAGE.items()})
        rows.append(row)
        print("  " + "  ".join(f"{a} {b:.3f}" if isinstance(b, (float, np.floating)) else f"{a} {b}"
                               for a, b in row.items()), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                     for r in rows])


def setup(args):
    """Training and held-out pools (the held-out features use the training stats) and the run tag."""
    dev = t.device("cuda")
    if args.encoder == "np" and args.feat != "raw":
        raise SystemExit("--encoder np reads the raw records; use --feat raw")
    fdev = t.device("cpu") if args.feat == "raw" else dev
    tr = load_pool(args.cache, "train", args.feat, dev, fdev=fdev)
    ho = [load_pool(args.cache, p, args.feat, dev, tr["stats"], fdev) for p in ("val", "test")]
    for D in [tr] + ho:
        D["origin"] = tr["y"].view(-1, 2).mean(0)
    tag = f"acoustic_{args.feat}" + (f"_{args.tag}" if args.tag else "") + (f"_s{args.seed}" if args.seed else "")
    return dev, tr, ho, tag


def evaluate(args):
    """Robustness of a trained regressor (--ckpt, built with the same flags) on the held-out pools."""
    dev, tr, ho, tag = setup(args)
    net = Net(args, tr["F"].shape[-1], tr["sp"].shape[1]).to(dev)
    net.load_state_dict(t.load(args.ckpt, map_location=dev)); net.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[{tag}] robustness of {args.ckpt}", flush=True)
    robustness(net, ho, args, os.path.join(args.out_dir, f"results_{tag}_robust.csv"))


def train(args):
    t.manual_seed(args.seed); np.random.seed(args.seed)
    dev, tr, ho, tag = setup(args)
    print(f"[{tag}] {tr['N'] * tr['ppt']} training points, features {tuple(tr['F'].shape[1:])}; "
          f"encoder {args.encoder}, positions {args.pos_code}, frame {args.frame}, aug {args.aug}, "
          f"loss {args.loss}, sensor drop {args.sensor_drop}, pos err {args.pos_err}", flush=True)

    net = Net(args, tr["F"].shape[-1], tr["sp"].shape[1]).to(dev)
    print(f"[{tag}] width {args.d}, layers {args.layers}, tokenizer {args.tokenizer}, "
          f"ffn {not args.no_ffn}, summary {args.summary}: "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = t.optim.AdamW(net.parameters(), args.lr, weight_decay=1e-4)
    sch = t.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    n_pts, t0, run = tr["N"] * tr["ppt"], time.time(), 0.0
    for i in range(1, args.steps + 1):
        idx = t.randint(n_pts, (args.batch,), device=dev)
        pos, F, tgt, _ = batch(tr, idx, args)
        mask = (drop_mask(len(idx), pos.shape[1], args.sensor_drop, args.min_keep, dev)
                if args.sensor_drop > 0 else None)
        with t.autocast("cuda", dtype=t.bfloat16):
            out = net(pos, F, mask).float()
        mu, var = gaussian(out, args)
        err = (mu - tgt).norm(dim=-1)
        loss = err.mean() if var is None else 0.5 * (var.log() + (tgt - mu) ** 2 / var).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        run += err.mean().item() if i % 50 == 0 else 0.0
        if i % args.log_every == 0:
            net.eval(); e = predict(net, ho[0], args, 1000)[0][:, 10:]; net.train()
            print(f"  step {i}: train {run / (args.log_every / 50) * SCALE:.2f} m, "
                  f"held-out (1000 transits) {e.mean():.2f} m, {time.time() - t0:.0f} s", flush=True)
            run = 0.0

    net.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    t.save(net.state_dict(), os.path.join(args.out_dir, f"{tag}.pt"))
    E = np.concatenate([predict(net, D, args)[0] for D in ho])       # (transits, ppt)
    G = np.concatenate([D["gid"] for D in ho])
    labels = {int(k): v.get("region", "?")
              for k, v in json.load(open(os.path.join(args.data_dir, "splits.json")))["labels"].items()}
    reg = np.array([labels.get(int(g), "?") for g in G])
    rows = [("all held-out", np.ones(len(G), bool))] + [(r, reg == r) for r in sorted(set(reg))]
    rows.append(("first 500 deployments", np.isin(G, np.sort(np.unique(G))[:500])))
    with open(os.path.join(args.out_dir, f"results_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "transits", "mae_m", "median_point_m", "p90_point_m", "within_5m", "mae_all50_m"])
        for name, m in rows:
            e = E[m, 10:]
            w.writerow([name, len(e), f"{e.mean():.3f}", f"{np.median(e):.3f}",
                        f"{np.percentile(e, 90):.2f}", f"{np.mean(e < 5):.3f}", f"{E[m].mean():.3f}"])
            print(f"[{tag}] {name}: {len(e)} transits, MAE {e.mean():.2f} m (all 50 points {E[m].mean():.2f}), "
                  f"median point {np.median(e):.2f} m, 90th pct {np.percentile(e, 90):.1f} m, "
                  f"within 5 m {np.mean(e < 5):.3f}", flush=True)
    print(f"[{tag}] robustness (all 50 points)", flush=True)
    robustness(net, ho, args, os.path.join(args.out_dir, f"results_{tag}_robust.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "train", "eval"])
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--feat", choices=["tone", "raw"], default="tone")
    ap.add_argument("--out-dir", default="output/eval_paper/baselines/b1")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--encoder", choices=["set", "np"], default="set")
    ap.add_argument("--tokenizer", choices=["mlp", "linear"], default="mlp",
                    help="set encoder: per-sensor token map (linear = the NPs' SpatialEncoder)")
    ap.add_argument("--no-ffn", action="store_true",
                    help="set encoder: attention-only mixing layers (as the NPs' SpatialEncoder)")
    ap.add_argument("--summary", choices=["readout", "pool"], default="readout",
                    help="set encoder: readout token through every layer, or one attention pool "
                         "after them (as the NPs' SpatialEncoder)")
    ap.add_argument("--pos-code", choices=["raw", "fourier"], default="raw")
    ap.add_argument("--frame", choices=["centroid", "absolute"], default="centroid")
    ap.add_argument("--aug", choices=["rot", "trans", "none"], default="rot")
    ap.add_argument("--trans-m", type=float, default=150.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--loss", choices=["euclid", "nll"], default="euclid",
                    help="nll: diagonal Gaussian output trained by negative log-likelihood")
    ap.add_argument("--var-floor", type=float, default=0.01, help="variance floor [m^2], --loss nll")
    ap.add_argument("--sensor-drop", type=float, default=0.0,
                    help="train-time probability of dropping each sensor (0 = off)")
    ap.add_argument("--min-keep", type=int, default=3, help="sensors always kept by --sensor-drop")
    ap.add_argument("--pos-err", type=float, default=0.0,
                    help="train-time sensor-position error: sd [m] ~ U(0, pos_err) per point (0 = off)")
    ap.add_argument("--ckpt", default=None, help="eval: state dict saved by train")
    A = ap.parse_args()
    {"extract": lambda: extract(A.data_dir, A.cache), "train": lambda: train(A),
     "eval": lambda: evaluate(A)}[A.stage]()
