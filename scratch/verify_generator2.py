"""Rigorous regression: OLD code (git HEAD) vs NEW code, identical seeds.
At default flags the impulse response must be BIT-IDENTICAL."""
import os, sys, importlib.util
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "data"))

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

NEW = load("data/acoustic_data_generator.py", "adg_new")
OLD = load("scratch/adg_BEFORE.py", "adg_old")
cfg = NEW.load_traj_config()

def gen(mod, seed=11, n_traj=3, ppt=50, df=50.0, B=None, abs_delay=False,
        sig='sinusoid', chirp_band=None):
    mod.channel.save_channel_info = lambda self, name: None
    np.random.seed(seed)
    p = mod.generate_params(options=0.1)
    if B is not None:
        p['ci']['B'] = float(B); p['ci']['fmax'] = p['ci']['fmin'] + p['ci']['B']
    p['ci']['df'] = df; p['n_traj'] = n_traj; p['ppt'] = ppt
    if abs_delay: p['absolute_delay'] = True
    if chirp_band: p['chirp_band'] = chirp_band
    np.random.seed(seed)
    tmp = mod.channel(load=False, params=dict(p), number_of_processes=1,
                      name="0.1", topology='random', traj_config=cfg)
    traj = tmp.traj
    rp = np.zeros((3, p['n_sensors'])); rp[0] = np.linspace(-200, 200, p['n_sensors']); rp[2] = 1.0
    orig = mod.channel.generate_sensor_positions
    mod.channel.generate_sensor_positions = lambda self, *a, **k: rp
    try:
        np.random.seed(seed + 5)
        c = mod.channel(load=False, params=dict(p), number_of_processes=1, name="0.1",
                        topology='random', precomputed_trajectories=traj, traj_config=cfg)
        np.random.seed(seed + 7)
        data, _ = c.filter(1024, snr=10.0, nt=n_traj, signal_type=sig, rep=1,
                           specific=list(range(n_traj)))
    finally:
        mod.channel.generate_sensor_positions = orig
    return np.asarray(c.h), np.asarray(data), rp, traj

h_old, d_old, _, _ = gen(OLD)
h_new, d_new, _, _ = gen(NEW)
print(f"OLD h {h_old.shape} | NEW h {h_new.shape}")
hb = np.array_equal(h_old, h_new); db = np.array_equal(d_old, d_new)
print(f"  impulse response bit-identical : {hb}")
print(f"  filtered_data   bit-identical : {db}")

# the flags must actually CHANGE things when switched on
h_ad, _, _, _   = gen(NEW, abs_delay=True)
_,   d_ch, _, _ = gen(NEW, sig='chirp')
print(f"\nflags have an effect when enabled:")
print(f"  absolute_delay=True changes h : {not np.array_equal(h_ad, h_new)} "
      f"(max|diff|={np.abs(h_ad-h_new).max():.4g})")
print(f"  signal_type=chirp changes data: {not np.array_equal(d_ch, d_new)}")
print(f"  |h| envelope UNCHANGED by absolute_delay (pure phase ramp): "
      f"{np.allclose(np.sort(np.abs(h_ad),axis=0), np.sort(np.abs(h_new),axis=0), atol=1e-6)}")
ok = hb and db
print("\n" + ("PASS: defaults are a no-op; flags are live"
              if ok else "FAIL: defaults changed behaviour"))
sys.exit(0 if ok else 1)
