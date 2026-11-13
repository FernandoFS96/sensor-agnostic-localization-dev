"""Restartable delay-feature preprocessing for the 10k-layout set.

The first attempt died silently ~1 h in with nothing written: no OOM, no
traceback, 941 GB disk free. Cause unattributable, so this version is built so
that a repeat costs one CHUNK instead of the whole run:

  * the split is frozen to splits.json FIRST, so a restart cannot reshuffle
    which layouts are train/val/test -- that would silently invalidate any
    comparison against runs made before the restart;
  * geometries are processed in chunks, each written to its own partial file,
    and an existing partial is SKIPPED on restart;
  * output is unbuffered (python -u) so progress and any error are on disk the
    moment they happen, rather than sitting in a block buffer that a kill throws
    away -- which is exactly why the first failure left no evidence.

Reads are disk-saturated (measured 214 MiB/s serial, 205 MiB/s with 8 threads),
so this stays SERIAL by design; parallelism would only add failure modes for no
throughput. Floor is ~90 min for the 1.1 TiB.

Sample construction is delegated to process_data.build_geometry_samples -- the
same function the original path uses -- so the on-disk format cannot drift from
what the trainer expects.
"""
import os, sys, json, pickle, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from process_data import (find_position_sets, set_index, assign_geometry_split,
                          build_geometry_samples, DELAY_FEATURE_NAMES)

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data/calib_10k/hermite_distinct")
ap.add_argument("--save-dir", default="data/calib_10k/hermite_distinct/processed_delay/geometry_split")
ap.add_argument("--part-dir", default="data/calib_10k/hermite_distinct/_parts")
ap.add_argument("--chunk", type=int, default=250)
ap.add_argument("--thetas", default="0.1")
ap.add_argument("--train-geoms", type=int, default=8000)
ap.add_argument("--val-geoms", type=int, default=1000)
ap.add_argument("--test-geoms", type=int, default=1000)
ap.add_argument("--shuffle-seed", type=int, default=42)
a = ap.parse_args()

DELAY = {"df": 0.5, "c": 1500.0, "roll": 50, "thresh": 0.30}
thetas = [float(x) for x in a.thetas.split(",")]
os.makedirs(a.save_dir, exist_ok=True)
os.makedirs(a.part_dir, exist_ok=True)

set_dirs = find_position_sets(a.data_root)
by_id = {set_index(sd): sd for sd in set_dirs}
print(f"geometries found: {len(set_dirs)}  thetas={thetas}", flush=True)

# ---- frozen split ---------------------------------------------------------
sp_path = os.path.join(a.save_dir, "splits.json")
if os.path.exists(sp_path):
    split = json.load(open(sp_path))
    split = {k: ([int(x) for x in v] if k in ("train", "val", "test") else v)
             for k, v in split.items()}
    print("split: reusing frozen splits.json", flush=True)
else:
    split = assign_geometry_split(set_dirs, thetas, a.train_geoms, a.val_geoms, a.test_geoms)
    json.dump(split, open(sp_path, "w"), indent=2)
    print("split: computed and frozen -> splits.json", flush=True)
for p in ("train", "val", "test"):
    print(f"  {p}: {len(split[p])} geometries", flush=True)

# ---- chunked build --------------------------------------------------------
meta = {"thetas": thetas, "split": split}
t_start = time.time()
for pool in ("val", "test", "train"):     # cheap pools first: fail fast, and a
    ids = list(split[pool])               # kill during train still leaves these
    chunks = [ids[i:i + a.chunk] for i in range(0, len(ids), a.chunk)]
    parts = []
    for ci, ch in enumerate(chunks):
        pf = os.path.join(a.part_dir, f"{pool}_{ci:04d}.pkl")
        parts.append(pf)
        if os.path.exists(pf):
            print(f"  [{pool} {ci+1}/{len(chunks)}] skip (exists)", flush=True)
            continue
        t0 = time.time()
        s, g, th, tau, ns = build_geometry_samples(by_id, ch, thetas, delay_cfg=DELAY)
        with open(pf + ".tmp", "wb") as f:          # atomic: a killed write
            pickle.dump({"s": s, "g": g, "th": th, "tau": int(tau), "ns": int(ns)}, f)
        os.replace(pf + ".tmp", pf)                 # never leaves a half file
        print(f"  [{pool} {ci+1}/{len(chunks)}] {len(s):,} samples "
              f"{time.time()-t0:.0f}s  (elapsed {(time.time()-t_start)/60:.0f}m)", flush=True)

    # ---- merge -------------------------------------------------------------
    samples, gids, ths, tau, ns = [], [], [], None, None
    for pf in parts:
        d = pickle.load(open(pf, "rb"))
        samples += d["s"]; gids += d["g"]; ths += d["th"]
        tau, ns = d["tau"], d["ns"]
    if pool == "train":     # same permutation the original applied
        rng = np.random.default_rng(a.shuffle_seed)
        perm = rng.permutation(len(samples))
        samples = [samples[i] for i in perm]
        gids = [gids[i] for i in perm]; ths = [ths[i] for i in perm]
    with open(os.path.join(a.save_dir, f"{pool}_data.pkl"), "wb") as f:
        pickle.dump(samples, f)
    feat_dim = int(np.asarray(samples[0]["X"]).shape[-1])
    meta[f"{pool}_geometry_ids"] = gids
    meta[f"{pool}_thetas"] = ths
    meta.update({"n_sensors": int(ns), "raw_tau": int(tau), "feat_dim": feat_dim,
                 "tau": feat_dim // int(ns), "features": "delay",
                 "delay_cfg": dict(DELAY),
                 "feature_names": list(DELAY_FEATURE_NAMES),
                 "m_per_tap": DELAY["c"] / (int(tau) * DELAY["df"])})
    print(f"  {pool}: WROTE {len(samples):,} samples X=(ppt,{feat_dim})", flush=True)
    del samples, gids, ths

with open(os.path.join(a.save_dir, "metadata.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"DONE in {(time.time()-t_start)/60:.0f} min -> {a.save_dir}", flush=True)
