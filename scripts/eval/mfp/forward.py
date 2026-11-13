"""Forward model for focalization: the generator's own physics (B2 settings, theta 0.1,
fixed seed, as b2_tables.py) evaluated only at given source points and sensors.
replica(env, P, S) -> complex tone (N, n_sensors) for sources P (N, 2) and sensors S (n, 2)."""
import os, sys, contextlib, io, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "data"))
_cwd = os.getcwd(); os.chdir(REPO)
import acoustic_data_generator as base
import random_position_generator as rand
os.chdir(_cwd)
W0 = np.pi / 20
# override_base_params wraps base.generate_params on every call; restore the original
# first, or a long-lived worker stacks ~1000 wrappers and hits the recursion limit.
_ORIG_GP = base.generate_params


def env_of(v):
    """(h0, hr0, c, c2_frac, k) -> environment dict, source at the surface (ht0 = h0)."""
    h0, hr0, c, fr, k = map(float, v)
    return {"h0": h0, "ht0": h0, "hr0": hr0, "c": c, "c2": c * fr, "k": k}


def replica(env, P, S, seed=11):
    P = np.asarray(P, float); S = np.asarray(S, float); N, n = len(P), len(S)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        base.generate_params = _ORIG_GP
        rand.override_base_params(df=50.0, n_traj=N, ppt=1)
        np.random.seed(seed)
        params = rand.apply_environment(base.generate_params(options=0.1), env)
        params["n_sensors"] = n
        rpos = np.zeros((3, n)); rpos[:2] = S.T; rpos[2] = params["ci"]["hr0"]
        traj = np.zeros((3, N, 2)); traj[:2, :, 0] = P.T; traj[:2, :, 1] = P.T
        orig = base.channel.generate_sensor_positions
        base.channel.generate_sensor_positions = lambda self, *a, **k: rpos
        try:
            c = base.channel(load=False, params=params, number_of_processes=1, name="0.1",
                             topology="random", precomputed_trajectories=traj, traj_config=None)
        finally:
            base.channel.generate_sensor_positions = orig
        D, _ = c.filter(1024, snr=200.0, nt=N, signal_type="sinusoid", rep=1, specific=list(range(N)))
    k = np.arange(D.shape[0])
    z = ((D[:, 0] * np.cos(W0 * k)[:, None, None]).sum(0)
         - 1j * (D[:, 0] * np.sin(W0 * k)[:, None, None]).sum(0)) * 2 / D.shape[0]
    return z  # (N, n)
