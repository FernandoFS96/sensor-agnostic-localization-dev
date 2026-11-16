"""
probe_frontend.py
=================

How much position information does the PFN's own front end carry WITHOUT any
context? The trained PFN's SensorSetEncoder is frozen; its per-point output
(num_hidden features) is computed for every training and held-out point, and a
small MLP is trained on the training points to regress the source position in
the array-centered frame. The probe sees one point at a time: no context.

  probe ~ regressor  -> the front end is fine; the PFN's point stack loses it
  probe ~ PFN        -> the front end inside the PFN is the limit

    python scratch/context_diag/probe_frontend.py --data-dir <split> --ckpt <run>/best.pt --tag b2_s0
"""
import argparse, os, pickle, sys, time
import numpy as np
import torch as t
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval")); sys.path.insert(0, REPO)
import eval_suite as E            # noqa: E402


@t.no_grad()
def encode(enc, data_dir, pool, dev, batch=256):
    S = pickle.load(open(os.path.join(data_dir, f"{pool}_data.pkl"), "rb"))
    H, Y, G = [], [], []
    for a in range(0, len(S), batch):
        ch = S[a:a + batch]
        X = t.as_tensor(np.stack([np.asarray(s["X"], np.float32) for s in ch]), device=dev)
        sp = t.as_tensor(np.stack([np.asarray(s["sensor_pos"], np.float32) for s in ch]), device=dev)
        y = t.as_tensor(np.stack([np.asarray(s["y"], np.float32)[:, :2] for s in ch]), device=dev)
        H.append(enc(X, sp).half())
        Y.append((y - sp[:, None, :, :2].mean(2)) / 100.0)             # array-centered, 100 m units
        G += [int(s["geometry_id"]) for s in ch]
    del S
    return t.cat(H), t.cat(Y), np.array(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "output/eval_paper/context_diag"))
    A = ap.parse_args()
    dev = t.device("cuda"); t.manual_seed(0)
    b = E.load_model(A.ckpt, dev)
    enc = b["model"].spatial_encoder
    t0 = time.time()
    Htr, Ytr, _ = encode(enc, A.data_dir, "train", dev)
    ho = [encode(enc, A.data_dir, p, dev) for p in ("val", "test")]
    Hho, Yho = t.cat([h[0] for h in ho]), t.cat([h[1] for h in ho])
    print(f"[{A.tag}] encoded {Htr.shape[0]} training and {Hho.shape[0]} held-out transits, "
          f"{Htr.shape[-1]} features, {time.time() - t0:.0f} s", flush=True)
    d = Htr.shape[-1]
    Htr, Ytr = Htr.reshape(-1, d), Ytr.reshape(-1, 2)
    mu, sd = Htr.float().mean(0), Htr.float().std(0) + 1e-6
    for name, net in (("linear", nn.Linear(d, 2)),
                      ("mlp", nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                            nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 2)))):
        net = net.to(dev)
        opt = t.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sch = t.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=A.steps, pct_start=0.05)
        for i in range(1, A.steps + 1):
            idx = t.randint(len(Htr), (A.batch,), device=dev)
            loss = (net((Htr[idx].float() - mu) / sd) - Ytr[idx]).norm(dim=-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            if i % 10000 == 0:
                print(f"  {name} step {i}: train {loss.item() * 100:.2f} m", flush=True)
        net.eval()
        with t.no_grad():
            pr = t.cat([net((Hho[a:a + 512].float() - mu) / sd) for a in range(0, len(Hho), 512)])
            e = ((pr - Yho).norm(dim=-1) * 100).cpu().numpy()                   # (transits, 50)
            rng = (Yho.norm(dim=-1) * 100).cpu().numpy()
            tr = t.cat([(net((Htr[a:a + 65536].float() - mu) / sd) - Ytr[a:a + 65536]).norm(dim=-1)
                        for a in range(0, 2_000_000, 65536)]).mean().item() * 100
        print(f"[{A.tag}] {name} probe: held-out MAE all50 {e.mean():.2f} m, late40 {e[:, 10:].mean():.2f}, "
              f"median point {np.median(e):.2f}, p90 {np.percentile(e, 90):.2f}; train points {tr:.2f} m", flush=True)
        print("    by source range from the array centroid: " + "  ".join(
            f"{lo}-{hi}: {e[(rng >= lo) & (rng < hi)].mean():.2f}"
            for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 10 ** 6))), flush=True)
        np.save(os.path.join(A.out_dir, f"probe_{A.tag}_{name}.npy"), e)


if __name__ == "__main__":
    main()
