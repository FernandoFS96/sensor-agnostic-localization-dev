"""Materialise one small data dir per (layouts, trajectories) cell.

TrajectoryDataset pickle.loads the WHOLE train file and the trainer applies
max_train_geometries only AFTER that, so a 100-layout run would otherwise pay
the full 400,000-sample / 4.0 GB resident cost to keep 1,000 samples -- and with
7 runs x num_workers=4 forking that list, the grid projected to ~83 GB against
116 GB available. Pre-subsetting makes each run resident in proportion to the
data it actually trains on.

Cells keep exactly the samples the trainer's own caps would have kept (first N
sorted geometry_ids, first M sorted traj_ids), so the run is identical to the
un-subsetted one -- the caps are left on the command line as no-op assertions.
Validation is capped to 10 trajectories ONCE and shared by every cell, so all
runs report MAE on an identical held-out set.
"""
import os, sys, pickle, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/calib_10k/hermite_distinct/processed_delay/geometry_split")
ap.add_argument("--out", default="data/calib_10k/cells")
ap.add_argument("--val-traj", type=int, default=10)
ap.add_argument("--cells", default="8000x50,8000x10,4000x10,1600x50,1600x10,400x10,100x10")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)

print("loading source split ...", flush=True)
train = pickle.load(open(f"{a.src}/train_data.pkl", "rb"))
val = pickle.load(open(f"{a.src}/val_data.pkl", "rb"))
meta = pickle.load(open(f"{a.src}/metadata.pkl", "rb"))
print(f"  train {len(train):,}  val {len(val):,}", flush=True)

# Shared, capped validation: identical metric across every cell.
vt = sorted({int(s["traj_id"]) for s in val})[:a.val_traj]
vset = set(vt)
val_c = [s for s in val if int(s["traj_id"]) in vset]
shared = os.path.join(a.out, "_shared")
os.makedirs(shared, exist_ok=True)
pickle.dump(val_c, open(f"{shared}/val_data.pkl", "wb"))
pickle.dump(meta, open(f"{shared}/metadata.pkl", "wb"))
print(f"  shared val: {len(val_c):,} samples "
      f"({len({int(s['geometry_id']) for s in val_c})} geometries x {len(vt)} traj)", flush=True)
del val

all_g = sorted({int(s["geometry_id"]) for s in train})
all_t = sorted({int(s["traj_id"]) for s in train})

for spec in a.cells.split(","):
    G, T = (int(x) for x in spec.split("x"))
    gset, tset = set(all_g[:G]), set(all_t[:T])
    sub = [s for s in train
           if int(s["geometry_id"]) in gset and int(s["traj_id"]) in tset]
    d = os.path.join(a.out, f"g{G}_t{T}")
    os.makedirs(d, exist_ok=True)
    pickle.dump(sub, open(f"{d}/train_data.pkl", "wb"))
    for f in ("val_data.pkl", "metadata.pkl"):        # one copy, shared
        link = os.path.join(d, f)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.relpath(f"{shared}/{f}", d), link)
    print(f"  {spec:>10}: {len(sub):>8,} samples  ~{len(sub)*10.4/1024:.2f} GB resident", flush=True)

print("cells ready ->", a.out, flush=True)
