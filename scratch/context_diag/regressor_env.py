"""
regressor_env.py
================

Upper bound on what any context can add on B2: the no-context regressor of
scripts/eval/acoustic_regressor.py, same recipe, GIVEN THE TRUE ENVIRONMENT of
the deployment. The environment (water depth h0, receiver height hr0, sound
speed c, bottom-speed ratio c2/c, spreading k; each mapped to [-1, 1] over the
B2 prior) is appended to every sensor token's features. The gap to the
environment-blind regressor is the most a calibration context could supply
through the environment.

--env-keys selects a subset (partial oracle); --env-shuffle gives each
deployment another deployment's environment (control: must not help).

    python scratch/context_diag/regressor_env.py --cache <cache> --seed 0 [--env-keys h0,c]
"""
import argparse, glob, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch as t

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval")); sys.path.insert(0, REPO)
import acoustic_regressor as R    # noqa: E402

DD = "data/pathD/hermite_distinct/processed/geometry_split"
PRIOR = {"h0": (30.0, 120.0), "hr0": (1.0, 5.0), "c": (1470.0, 1540.0), "c2f": (0.70, 0.95), "k": (1.4, 2.0)}


def env_table(keys, shuffle):
    env = {}
    for f in glob.glob(os.path.join(REPO, "data/pathD/hermite_distinct/_manifest_*.pkl")):
        env.update(pickle.load(open(f, "rb"))["environments"])
    n = max(env) + 1
    T = np.zeros((n, len(keys)), np.float32)
    for g, e in env.items():
        v = dict(e, c2f=e["c2"] / e["c"])
        T[g] = [(v[k] - sum(PRIOR[k]) / 2) / ((PRIOR[k][1] - PRIOR[k][0]) / 2) for k in keys]
    if shuffle:
        T = T[np.random.default_rng(0).permutation(n)]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--env-keys", default="h0,hr0,c,c2f,k")
    ap.add_argument("--env-shuffle", action="store_true")
    ap.add_argument("--tag", default="nll_drop_env")
    a = ap.parse_args()
    keys = a.env_keys.split(",")
    T = env_table(keys, a.env_shuffle)
    orig = R.load_pool

    def load_pool(cache, pool, feat, dev, stats=None, fdev=None):
        D = orig(cache, pool, feat, dev, stats, fdev)
        e = t.as_tensor(T[D["gid"]], dtype=D["F"].dtype, device=D["F"].device)        # (N, K)
        ns = D["F"].shape[1]
        e = e[:, None, None, :].expand(D["N"], D["ppt"], ns, len(keys)).reshape(D["N"] * D["ppt"], ns, len(keys))
        D["F"] = t.cat([D["F"], e], -1)
        return D
    R.load_pool = load_pool
    args = SimpleNamespace(
        stage="train", data_dir=os.path.join(REPO, DD), cache=a.cache, feat="tone",
        out_dir=os.path.join(REPO, "output/eval_paper/baselines/b2"), steps=a.steps, batch=2048, lr=3e-4,
        d=256, layers=6, seed=a.seed, log_every=5000, encoder="set", tokenizer="mlp", no_ffn=False,
        summary="readout", pos_code="raw", frame="centroid", aug="rot", trans_m=150.0, tag=a.tag,
        loss="nll", var_floor=0.01, sensor_drop=0.3, min_keep=3, pos_err=0.0, ckpt=None)
    print(f"[env] keys {keys}, shuffle {a.env_shuffle}", flush=True)
    R.train(args)


if __name__ == "__main__":
    main()
