"""The two variants may differ in exactly ONE factor: the trajectories.

Four checks, run before spending 30 h of GPU:
  1. identical geometry split (exact ids per split, not just counts)
  2. identical interp/extrap labels and centre -> per-region MAE is comparable
  3. same sample counts, same feature dimension, same thetas
  4. that in `distinct` the trajectories DO change between deployments, and in
     `shared` they do NOT
Check 4 is what gives the experiment meaning: without it the A/B compares two
copies of the same thing.
"""
import json, pickle, glob, re
import numpy as np

SH = "data/random_task/hermite_shared/processed/geometry_split"
DI = "data/random_task/hermite_distinct/processed_common/geometry_split"

ok = True
def chk(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}{(' -- ' + extra) if extra else ''}")

print("\n=== 1-2. split and labels ===")
a, b = (json.load(open(f"{d}/splits.json")) for d in (SH, DI))
for k in ("train", "val", "test"):
    chk(f"{k}: same {len(a[k])} ids", sorted(a[k]) == sorted(b[k]),
        f"shared={len(a[k])} distinct={len(b[k])}")
chk("centre of centroids identical",
    np.allclose(a["center_of_centroids"], b["center_of_centroids"]))
chk("interp/extrap labels identical",
    {k: v["region"] for k, v in a["labels"].items()} ==
    {k: v["region"] for k, v in b["labels"].items()})
chk("centroids identical (same layouts)",
    all(np.allclose(a["labels"][k]["centroid"], b["labels"][k]["centroid"])
        for k in a["labels"]))

print("\n=== 3. samples, features, thetas ===")
ma, mb = (pickle.load(open(f"{d}/metadata.pkl", "rb")) for d in (SH, DI))
for k in ("train", "val", "test"):
    na, nb = len(ma[f"{k}_geometry_ids"]), len(mb[f"{k}_geometry_ids"])
    chk(f"{k}: {na:,} samples", na == nb, f"shared={na:,} distinct={nb:,}")
for k in ("tau", "n_sensors", "feat_dim"):
    chk(f"{k} = {ma[k]}", ma[k] == mb[k], f"shared={ma[k]} distinct={mb[k]}")
chk(f"thetas = {ma['thetas']}", sorted(ma["thetas"]) == sorted(mb["thetas"]),
    f"shared={ma['thetas']} distinct={mb['thetas']}")

print("\n=== 4. do trajectories change per deployment? ===")
for name, root in (("shared  ", "data/random_task/hermite_shared"),
                   ("distinct", "data/random_task/hermite_distinct")):
    # One file per deployment: each position_set has one directory per theta and
    # both share trajectories (theta is a channel factor, not a trajectory one), so
    # the raw glob would count every deployment twice and flag a false duplicate.
    seen, f = set(), []
    for x in sorted(glob.glob(f"{root}/position_set_*/**/trajectories.npy", recursive=True)):
        sid = re.search(r"position_set_(\d+)", x).group(1)
        if sid not in seen:
            seen.add(sid); f.append(x)
    f = f[:40]
    if not f:
        print(f"  [ -- ] {name}: no trajectories.npy found")
        continue
    t0 = np.load(f[0])
    same = sum(np.allclose(np.load(x), t0) for x in f[1:])
    tot = len(f) - 1
    esperado_igual = name.strip() == "shared"
    chk(f"{name}: {same}/{tot} deployments with trajectories identical to the first",
        (same == tot) if esperado_igual else (same == 0),
        "expected: all identical" if esperado_igual else "expected: none identical")

print(f"\n=== VERDICT: {'exactly one factor changes, A/B valid' if ok else 'REVIEW -- more than one factor differs'} ===")
raise SystemExit(0 if ok else 1)
