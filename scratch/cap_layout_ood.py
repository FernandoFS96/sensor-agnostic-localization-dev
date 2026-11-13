"""Materialise the CAPPED layout_ood split once, so runs stop loading 51.5 GB.

The reference spatial_cnp run (23.50 m) trains on max_train_geometries=140 and
max_train_trajectories=350, but TrajectoryDataset pickle.loads the whole
train_data.pkl (51.5 GB / 128,000 samples at 402 KB each) and the trainer caps
only afterwards. One process therefore needs ~58 GB resident to train on ~38% of
it, which makes concurrent seeds impossible on a 125 GB box.

Subsetting keeps exactly what the trainer's own caps would keep -- first N
sorted geometry_ids, first M sorted traj_ids -- so a run on this directory is
identical to one on the full split. The caps are left on the command line as
no-op assertions.
"""
import os, pickle, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/random_task/hermite_shared/processed/geometry_split")
ap.add_argument("--out", default="data/layout_ood_capped")
ap.add_argument("--geoms", type=int, default=140)
ap.add_argument("--traj", type=int, default=350)
ap.add_argument("--val-traj", type=int, default=350)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)

def cap(split, n_geom, n_traj):
    print(f"  loading {split} ...", flush=True)
    s = pickle.load(open(f"{a.src}/{split}_data.pkl", "rb"))
    g_all = sorted({int(r["geometry_id"]) for r in s})
    t_all = sorted({int(r["traj_id"]) for r in s})
    gs = set(g_all[:n_geom]) if n_geom else set(g_all)
    ts = set(t_all[:n_traj]) if n_traj else set(t_all)
    sub = [r for r in s
           if int(r["geometry_id"]) in gs and int(r["traj_id"]) in ts]
    del s
    pickle.dump(sub, open(f"{a.out}/{split}_data.pkl", "wb"), protocol=4)
    print(f"  {split}: {len(sub):,} samples "
          f"({len(gs)} geoms x {len(ts)} traj)  ~{len(sub)*402/1e6:.1f} GB", flush=True)
    del sub

# Val/test geometries are the held-out layouts and must NOT be dropped; only
# their trajectory count is capped, matching max_val_trajectories.
cap("train", a.geoms, a.traj)
cap("val", None, a.val_traj)

meta = pickle.load(open(f"{a.src}/metadata.pkl", "rb"))
pickle.dump(meta, open(f"{a.out}/metadata.pkl", "wb"), protocol=4)
for extra in ("splits.json",):
    p = f"{a.src}/{extra}"
    if os.path.exists(p):
        open(f"{a.out}/{extra}", "wb").write(open(p, "rb").read())
print("capped split ready ->", a.out, flush=True)
