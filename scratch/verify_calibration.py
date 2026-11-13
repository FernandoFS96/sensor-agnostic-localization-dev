"""Verification for the CALIBRATION collate change.

PART A  regression: with calibration=false, the NEW collate must produce output
        BIT-IDENTICAL to the snapshot taken before the edit (scratch/
        train_np_geometry_BEFORE.py), for every pre-existing code path.
PART B  correctness: calibration=true does exactly what we intend.
"""
import os, sys, importlib.util, pickle
import numpy as np, torch as t
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "train"))

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

NEW = load("scripts/train/train_np_geometry.py", "trainer_new")
OLD = load("scratch/train_np_geometry_BEFORE.py", "trainer_old")

DD = "data/random_task/hermite_shared/processed/geometry_split"
ds = NEW.TrajectoryDataset(os.path.join(DD, "val_data.pkl"))
# keep it small & fast but multi-geometry
keep_g = sorted({int(s["geometry_id"]) for s in ds.samples})[:4]
ds.samples = [s for s in ds.samples if int(s["geometry_id"]) in keep_g][:600]
print(f"probe pool: {len(ds.samples)} samples over geometries {keep_g}, ppt={ds.ppt}")

def batch_of(idxs):  return [ds[i] for i in idxs]
IDX = [0, 1, 2, 3, 250, 251, 400, 401]

def same(a, b):
    if a is None or b is None: return a is None and b is None
    return t.equal(a, b) if isinstance(a, t.Tensor) else a == b

fails = []
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond: fails.append(msg)

# ---------------- PART A: regression, calibration OFF ----------------
print("\n=== PART A: backward compatibility (calibration=false) ===")
CASES = [
    dict(name="train per_sample=True  aug=0",  kw=dict(per_sample=True,  aug_m=0.0)),
    dict(name="train per_sample=True  aug=50", kw=dict(per_sample=True,  aug_m=50.0)),
    dict(name="train per_sample=False aug=0",  kw=dict(per_sample=False, aug_m=0.0)),
    dict(name="train per_sample=False aug=50", kw=dict(per_sample=False, aug_m=50.0)),
    dict(name="val   fixed_ctx=10",            kw=dict(fixed_ctx=10)),
    dict(name="val   fixed_ctx=10 mode=random",kw=dict(fixed_ctx=10, ctx_sample_mode="random")),
    dict(name="train excl_from_target=False",  kw=dict(per_sample=False, exclude_ctx_from_target=False)),
]
for c in CASES:
    kw = dict(ctx_sample_mode="first", exclude_ctx_from_target=True); kw.update(c["kw"])
    co = OLD.make_collate(2, 40, seed=123, **kw)
    cn = NEW.make_collate(2, 40, seed=123, **kw)   # calibration defaults False
    bo, bn = co(batch_of(IDX)), cn(batch_of(IDX))
    ok = set(bo.keys()) == set(bn.keys()) and all(same(bo[k], bn[k]) for k in bo)
    check(ok, f"{c['name']}: keys+tensors identical to pre-change code "
              f"({len(bo)} keys: {sorted(bo.keys())[:4]}...)")

# ---------------- PART B: calibration ON ----------------
print("\n=== PART B: calibration=true correctness ===")
by = {}
for i, s in enumerate(ds.samples):
    by.setdefault((int(s["geometry_id"]), round(float(s["theta"]), 6)), []).append(i)

cal = NEW.make_collate(2, 40, seed=7, ctx_sample_mode="first",
                       exclude_ctx_from_target=True, per_sample=True,
                       aug_m=0.0, calibration=True, dataset=ds)
b = cal(batch_of(IDX))
B, ppt = len(IDX), ds.ppt
check(set(b.keys()) == {"context_x","context_y","context_mask","target_x","target_y",
                        "target_mask","gids","thetas","sensor_pos"},
      f"emits only split-convention keys (no x_seq/context_indices): {sorted(b.keys())}")
check(b["target_x"].shape == (B, ppt, ds.feat_dim) and b["target_mask"].all(),
      f"targets = ALL {ppt} points of own trajectory, all scored")

# B1: every context point must belong to a DIFFERENT trajectory, same geometry+theta
ymap = {}   # y-row tuple -> set of (gid, theta, traj_id)
for s in ds.samples:
    for row in np.asarray(s["y"], np.float32):
        ymap.setdefault(row.tobytes(), set()).add(
            (int(s["geometry_id"]), round(float(s["theta"]),6), int(s.get("traj_id",-1))))

bad_geom = bad_self = overlap = 0
for i in range(B):
    own = ds.samples[IDX[i]]
    own_key = (int(own["geometry_id"]), round(float(own["theta"]),6), int(own.get("traj_id",-1)))
    own_rows = {r.tobytes() for r in np.asarray(own["y"], np.float32)}
    k = int(b["context_mask"][i].sum())
    for j in range(k):
        row = b["context_y"][i, j].numpy().astype(np.float32).tobytes()
        srcs = ymap.get(row, set())
        if not any(s[0] == own_key[0] and s[1] == own_key[1] for s in srcs): bad_geom += 1
        if row in own_rows: overlap += 1
        if srcs and all(s == own_key for s in srcs): bad_self += 1
check(bad_geom == 0, f"all context points come from the SAME (geometry, theta): {bad_geom} violations")
check(bad_self == 0, f"donor is never the target's OWN trajectory: {bad_self} violations")
check(overlap == 0, f"context and target points are DISJOINT: {overlap} overlapping points")

# B2: sensor_pos invariant -- donor shares the target's array layout
sp = b["sensor_pos"]
sp_ok = all(np.array_equal(np.asarray(ds.samples[IDX[i]]["sensor_pos"], np.float32), sp[i].numpy())
            for i in range(B))
check(sp_ok, "sensor_pos is the target sample's layout (donor shares it by construction)")

# B3: context sizes honour ctx_min/ctx_max, and fixed_ctx for validation
ks = [int(b["context_mask"][i].sum()) for i in range(B)]
check(all(2 <= k <= 40 for k in ks) and len(set(ks)) > 1,
      f"per-sample context sizes within [2,40] and varied: {ks}")
calv = NEW.make_collate(2, 40, fixed_ctx=10, seed=9, ctx_sample_mode="first",
                        calibration=True, dataset=ds)
bv = calv(batch_of(IDX))
check(all(int(bv["context_mask"][i].sum()) == 10 for i in range(B)),
      "validation fixed_ctx=10 honoured under calibration")

# B4: spatial separation -- context track really is elsewhere
d_ctx_tgt = []
for i in range(B):
    k = int(b["context_mask"][i].sum())
    c = b["context_y"][i, :k, :2].numpy(); tg = b["target_y"][i, :, :2].numpy()
    d_ctx_tgt.append(np.linalg.norm(c[:, None, :] - tg[None, :, :], axis=-1).min(axis=1).mean())
check(float(np.mean(d_ctx_tgt)) > 10.0,
      f"context track is spatially distinct from target track (mean nearest-point "
      f"distance {np.mean(d_ctx_tgt):.1f} m)")

# B5: augmentation applies ONE rigid shift to context y, target y and sensor_pos
cal_a = NEW.make_collate(2, 40, fixed_ctx=10, seed=11, ctx_sample_mode="first",
                         aug_m=200.0, calibration=True, dataset=ds)
cal_n = NEW.make_collate(2, 40, fixed_ctx=10, seed=11, ctx_sample_mode="first",
                         aug_m=0.0, calibration=True, dataset=ds)
ba, bn_ = cal_a(batch_of(IDX)), cal_n(batch_of(IDX))
d_t = (ba["target_y"][..., :2] - bn_["target_y"][..., :2])
d_s = (ba["sensor_pos"][..., :2] - bn_["sensor_pos"][..., :2])
m = bn_["context_mask"]
d_c = (ba["context_y"][..., :2] - bn_["context_y"][..., :2])
per = d_t[:, 0, :]                                   # claimed per-sample delta
ok_t = t.allclose(d_t, per[:, None, :], atol=1e-4)
ok_s = t.allclose(d_s, per[:, None, :], atol=1e-4)
ok_c = all(t.allclose(d_c[i][m[i]], per[i][None, :].expand(int(m[i].sum()), 2), atol=1e-4)
           for i in range(B))
check(ok_t and ok_s and ok_c and per.abs().max() > 1.0,
      f"aug applies ONE rigid shift to target_y, sensor_pos AND context_y "
      f"(|delta| up to {per.abs().max():.0f} m) -> one frame")
check(t.equal(ba["context_x"], bn_["context_x"]) and t.equal(ba["target_x"], bn_["target_x"]),
      "aug leaves acoustics X untouched (channel depends only on relative geometry)")

# B6: guard -- singleton donor group must fail loudly
solo = NEW.TrajectoryDataset(os.path.join(DD, "val_data.pkl"))
seen, uniq = set(), []
for s in solo.samples:
    k = (int(s["geometry_id"]), round(float(s["theta"]),6))
    if k not in seen: seen.add(k); uniq.append(s)
solo.samples = uniq
try:
    NEW.make_collate(2, 40, calibration=True, dataset=solo); ok = False
except ValueError as e:
    ok = "at least" in str(e) or ">=2" in str(e)
check(ok, "raises loudly when a (geometry, theta) group has <2 trajectories")
try:
    NEW.make_collate(2, 40, calibration=True, dataset=None); ok2 = False
except ValueError: ok2 = True
check(ok2, "raises when calibration=True without a dataset")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
