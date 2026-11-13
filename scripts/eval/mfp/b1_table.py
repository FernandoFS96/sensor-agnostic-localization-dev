"""Replica table for B1 MFP: complex tone amplitude against source-sensor distance,
from the generator's own physics (B1 settings), for theta 0.1 and 0.2, averaged
over 3 realizations of the random small-scale channel. 10 sensors at x-offsets of
0.1 m and sources at 1 m steps give a 0.1 m distance grid in one simulator call.

    python scripts/eval/mfp/b1_table.py --out b1_table_exact.npz
    python scripts/eval/mfp/b1_table.py --dc 15 --out b1_table_c15.npz        # model errors
    python scripts/eval/mfp/b1_table.py --dh 2.5 --out b1_table_h25.npz
    python scripts/eval/mfp/b1_table.py --dc 15 --dh 2.5 --out b1_table_c15h25.npz"""
import os, sys, argparse, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(REPO, "output", "eval_paper", "mfp", "cache")
os.chdir(REPO); sys.path.insert(0, "data")
import acoustic_data_generator as base
import random_position_generator as rand

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="b1_table_exact.npz", help="file name in --work, or a path")
ap.add_argument("--work", default=WORK)
ap.add_argument("--dc", type=float, default=0.0, help="sound-speed offset [m/s]")
ap.add_argument("--dh", type=float, default=0.0, help="water-depth offset [m] (source stays at the surface)")
A = ap.parse_args(); OUT = os.path.join(A.work, A.out)
xs = np.arange(1.0, 1301.0, 1.0)                       # source x positions [m]
offs = np.arange(10) * 0.1                             # sensor x offsets [m]

def run(theta, seed):
    N = len(xs)
    rand.override_base_params(df=50.0, n_traj=N, ppt=1)
    np.random.seed(seed)
    params = base.generate_params(options=theta)
    params['ci']['c'] += A.dc; params['ci']['h0'] += A.dh; params['ci']['ht0'] += A.dh
    rpos = np.zeros((3, 10)); rpos[0] = offs; rpos[2] = params['ci']['hr0']
    traj = np.zeros((3, N, 2)); traj[0, :, 0] = xs; traj[0, :, 1] = xs   # at rest
    orig = base.channel.generate_sensor_positions
    base.channel.generate_sensor_positions = lambda self, *a, **k: rpos
    try:
        c = base.channel(load=False, params=params, number_of_processes=10, name=str(theta),
                         topology="random", precomputed_trajectories=traj, traj_config=None)
    finally:
        base.channel.generate_sensor_positions = orig
    D, _ = c.filter(1024, snr=200.0, nt=N, signal_type="sinusoid", rep=1, specific=list(range(N)))
    n = np.arange(D.shape[0]); w0 = np.pi / 20
    z = ((D[:, 0] * np.cos(w0 * n)[:, None, None]).sum(0)
         - 1j * (D[:, 0] * np.sin(w0 * n)[:, None, None]).sum(0)) * 2 / D.shape[0]   # (N, 10)
    d = np.sqrt((xs[:, None] - offs[None]) ** 2 + params['ci']['hr0'] ** 2)
    return d.ravel(), z.ravel()

res = {}
for th in (0.1, 0.2):
    Z = []
    for seed in (11, 12, 13):
        d, z = run(th, seed); Z.append(z)
        print(f"theta {th} seed {seed} done", flush=True)
    o = np.argsort(d); Z = np.stack(Z)[:, o]
    res[f"d_{th}"] = d[o]; res[f"z_{th}"] = Z.mean(0); res[f"zs_{th}"] = Z
    la = np.log(np.abs(Z)); print(f"  theta {th}: seed-to-seed sd of log amplitude {la.std(0).mean():.3f}", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **res)
print("saved", OUT)
