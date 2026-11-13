"""Do the models actually USE the acoustics, or extrapolate the context geometry?

The layout_ood split has calibration=false, so context is an ordered PREFIX of
the target's own trajectory and the model is handed context_y -- the true
positions of those points. make_collate's own docstring flags the consequence:
"with a prefix of the target's OWN path revealed, a model can score well by
extrapolating that path and ignoring the acoustics entirely."

A constant-velocity baseline (83.67 m) does NOT rule this out: a learned model
can follow the hermite manifold far better than a straight line while still
never reading the acoustics. The only decisive test is to break the acoustics
and see whether the prediction survives.

Four conditions, all on the identical frozen val batches:
  intact          reference; must reproduce the reported MAE
  scramble        acoustics permuted ACROSS samples -- x_seq no longer describes
                  this trajectory, while context positions and timing are intact.
                  A model that truly localizes must collapse toward chance.
  noise           acoustics replaced by Gaussian noise of matched scale.
  shuffle_time    the temporal ORDER of the sequence is permuted (acoustics and
                  labels kept aligned). Isolates reliance on temporal continuity
                  rather than on the acoustics themselves.
  ctx2            context cut from 10 points to 2. Kills extrapolation (too few
                  points to fit a path) while leaving acoustics fully available.
"""
import sys, glob, pickle, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts/train")
import numpy as np, torch as t
from torch.utils.data import DataLoader
from scripts.train.train_np_geometry import (TrajectoryDataset, make_collate,
                                             compute_y_stats)
sys.path.insert(0, "scripts/eval")
from eval_suite import load_model   # reads num_hidden / readout / attn_ffn from
                                    # the checkpoint config, so each arm is
                                    # rebuilt exactly as it was trained

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/pfn10k/hermite_shared/processed/geometry_split")
ap.add_argument("--batches", type=int, default=24)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--tags", default=None,
                help="comma-separated exp_name:arch pairs; default = the FAM_ set. "
                     "An exp_name containing 'cnpF' is built with attn_ffn=True.")
a = ap.parse_args()

SPATIAL = dict(enabled=True, tokenize=True, use_position=True, use_attention=True,
               pooling='attention', norm_acoustic='standardize', n_attn_layers=3,
               n_fourier_bands=8, min_wavelength=10.0, max_wavelength=1000.0,
               pos_dim=2, sensor_id_embed=False)

dev = "cuda" if t.cuda.is_available() else "cpu"
train_ds = TrajectoryDataset(f"{a.data}/train_data.pkl")
y_mean, y_std = compute_y_stats(train_ds)
del train_ds
val_ds = TrajectoryDataset(f"{a.data}/val_data.pkl")
feat_dim = int(np.asarray(val_ds.samples[0]["X"]).shape[-1])

def batches(fixed_ctx):
    coll = make_collate(2, 40, fixed_ctx=fixed_ctx, seed=1234,
                        ctx_sample_mode="first", exclude_ctx_from_target=True)
    dl = DataLoader(val_ds, batch_size=a.bs, shuffle=False, num_workers=0, collate_fn=coll)
    out = []
    for i, b in enumerate(dl):
        if i >= a.batches: break
        out.append(b)
    return out

def perturb(b, mode, g):
    b = dict(b)
    if mode == "scramble":                       # acoustics from a DIFFERENT sample
        b["x_seq"] = b["x_seq"][t.randperm(b["x_seq"].shape[0], generator=g)]
    elif mode == "noise":
        x = b["x_seq"]; b["x_seq"] = t.randn(x.shape, generator=g) * x.std() + x.mean()
    elif mode == "shuffle_time":                 # break temporal order, keep pairing
        p = t.randperm(b["x_seq"].shape[1], generator=g)
        b["x_seq"] = b["x_seq"][:, p, :]; b["y_full"] = b["y_full"][:, p, :]
    ci, ti = b["context_indices"], b["target_indices"]
    b["context_x"] = b["x_seq"][:, ci, :]; b["context_y"] = b["y_full"][:, ci, :]
    b["target_x"]  = b["x_seq"][:, ti, :]; b["target_y"]  = b["y_full"][:, ti, :]
    return b

@t.no_grad()
def evaluate(model, conv, bs, mode, seed=0):
    g = t.Generator().manual_seed(seed)
    model.eval(); ym, ys = y_mean.to(dev), y_std.to(dev)
    tot, n = 0.0, 0
    for b in bs:
        b = perturb(b, mode, g)
        cy = ((b["context_y"].to(dev) - ym) / ys)
        ty = ((b["target_y"].to(dev) - ym) / ys)
        sp = b["sensor_pos"].to(dev) if b["sensor_pos"] is not None else None
        if conv == "split":
            out = model(b["context_x"].to(dev), cy, b["target_x"].to(dev), ty,
                        1.0, sensor_pos=sp)
        else:
            out = model(b["x_seq"].to(dev), b["context_indices"].to(dev), cy,
                        b["target_indices"].to(dev), ty, 1.0, sensor_pos=sp)
        pred = out[0] * ys + ym
        tot += float((pred - b["target_y"].to(dev)).abs().mean()) * ty.shape[0]
        n += ty.shape[0]
    return tot / max(n, 1)

if a.tags:
    MODELS = [(tg.split(":")[0], tg.split(":")[1], tg.split(":")[0])
              for tg in a.tags.split(",") if tg.strip()]
else:
    MODELS = [("FAM_ranp", "ranp", "spatial_RANP"), ("FAM_rcnp", "rcnp", "spatial_RCNP"),
              ("FAM_pfn", "pfn", "spatial_PFN(joint)"), ("FAM_cnpF", "cnp", "spatial_CNP+FFN"),
              ("FAM_cnp", "cnp", "spatial_CNP")]
CONDS = ["intact", "scramble", "noise", "shuffle_time"]

bs10 = batches(10)
bs2 = batches(2)
print(f"\n  ACOUSTIC-DEPENDENCE ABLATION  ({a.batches} frozen val batches, ctx=10)\n")
print(f"  {'model':<20}" + "".join(f"{c:>13}" for c in CONDS) + f"{'ctx=2':>10}")
for tag, arch, lbl in MODELS:
    ck = sorted(glob.glob(f"output/hydra/training/*/{tag}/seed_0/best.pt"))
    if not ck:
        print(f"  {lbl:<20}  (no checkpoint)"); continue
    bundle = load_model(ck[-1], dev)
    m, conv = bundle["model"], bundle["conv"]
    # x_mean / x_scale are registered BUFFERS restored by load_state_dict --
    # overwriting them here would destroy the input scaling of every model that
    # actually reads the acoustics (and silently spare the one that does not).
    row = [evaluate(m, conv, bs10, c) for c in CONDS]
    row.append(evaluate(m, conv, bs2, "intact"))
    print(f"  {lbl:<20}" + "".join(f"{v:>13.2f}" for v in row))
    del m
print("\n  A model that genuinely localizes should DEGRADE BADLY under scramble/noise.")
print("  One that extrapolates the context path will barely move.")
