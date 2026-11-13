"""B2 replica tables: complex tone amplitude against HORIZONTAL source-sensor distance,
from the generator's own physics (B2 settings, theta 0.1), one table per environment.
  --nominal       one table for the centre of B2's environment prior -> b2_nominal.npz
  (default)       one table per held-out deployment, with its true environment -> b2_true.npz"""
import os, sys, glob, json, pickle, argparse, numpy as np
from multiprocessing import Pool
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(REPO, "output", "eval_paper", "mfp", "cache")
B2 = os.path.join(REPO, "data", "pathD", "hermite_distinct")
xs = np.arange(1.0, 1301.0, 1.0); offs = np.arange(10) * 0.1
NOMINAL = {"h0": 75.0, "ht0": 75.0, "hr0": 3.0, "c": 1505.0, "c2": 1505.0 * 0.825, "k": 1.7}

def table(env, seed=11, nproc=10):
    os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "data"))
    import acoustic_data_generator as base
    import random_position_generator as rand
    N = len(xs)
    rand.override_base_params(df=50.0, n_traj=N, ppt=1)
    np.random.seed(seed)
    params = rand.apply_environment(base.generate_params(options=0.1), env)
    rpos = np.zeros((3, 10)); rpos[0] = offs; rpos[2] = params['ci']['hr0']
    traj = np.zeros((3, N, 2)); traj[0, :, 0] = xs; traj[0, :, 1] = xs
    orig = base.channel.generate_sensor_positions
    base.channel.generate_sensor_positions = lambda self, *a, **k: rpos
    try:
        c = base.channel(load=False, params=params, number_of_processes=nproc, name="0.1",
                         topology="random", precomputed_trajectories=traj, traj_config=None)
    finally:
        base.channel.generate_sensor_positions = orig
    D, _ = c.filter(1024, snr=200.0, nt=N, signal_type="sinusoid", rep=1, specific=list(range(N)))
    n = np.arange(D.shape[0]); w0 = np.pi / 20
    z = ((D[:, 0] * np.cos(w0 * n)[:, None, None]).sum(0)
         - 1j * (D[:, 0] * np.sin(w0 * n)[:, None, None]).sum(0)) * 2 / D.shape[0]
    h = (xs[:, None] - offs[None]).ravel(); o = np.argsort(h)
    return h[o], z.ravel()[o].astype(np.complex64)

def job(item):
    g, env = item
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        h, z = table(env, nproc=1)
    return g, h, z

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--nominal", action="store_true")
    ap.add_argument("--work", default=WORK); ap.add_argument("--procs", type=int, default=48)
    A = ap.parse_args()
    os.makedirs(A.work, exist_ok=True)
    if A.nominal:
        g, h, z = job((-1, NOMINAL))
        np.savez(os.path.join(A.work, "b2_nominal.npz"), h=h, z=z); print("nominal table saved")
        sys.exit()
    envs = {}
    for f in sorted(glob.glob(f"{B2}/_manifest*.pkl")):
        envs.update(pickle.load(open(f, "rb"))["environments"])
    sp = json.load(open(f"{B2}/processed/geometry_split/splits.json"))
    gids = sorted(set(sp["val"]) | set(sp["test"]))
    out = {}
    with Pool(A.procs) as pool:
        for i, (g, h, z) in enumerate(pool.imap_unordered(job, [(g, envs[g]) for g in gids])):
            out[g] = z
            if i % 50 == 0: print(f"  {i+1}/{len(gids)} tables", flush=True)
    np.savez(os.path.join(A.work, "b2_true.npz"), h=h, gids=np.array(sorted(out)), z=np.stack([out[g] for g in sorted(out)]))
    print("true-environment tables saved", flush=True)
