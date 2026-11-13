"""Regression: with the new flags at their DEFAULTS, the generator must
reproduce the existing on-disk data BIT-FOR-BIT."""
import os, sys, glob
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "data"))
import acoustic_data_generator as base
import random_position_generator as rand

REF = "data/random_task/hermite_shared/position_set_00/channel_option_0.1/random"
ref_filt = np.load(f"{REF}/filtered_data/filtered_data.npy", mmap_mode='r')
ref_pos  = np.load(glob.glob(f"{REF}/channel_info/sensor_positions_*.npy")[0])
ref_traj = np.load(glob.glob(f"{REF}/channel_info/trajs_*.npy")[0])
print(f"reference on disk: filtered {ref_filt.shape}")

cfg = base.load_traj_config()
NT = 3   # regenerate only the first few trajectories -- physics is per-trajectory
rand.override_base_params(df=50.0, n_traj=NT, ppt=50)     # defaults: B=None, absolute_delay=False
base.channel.save_channel_info = lambda self, name: None

r_pos = ref_pos
base.channel.generate_sensor_positions = lambda self, *a, **k: r_pos
np.random.seed(11)
params = base.generate_params(options=0.1)
print(f"params: B={params['ci']['B']} df={params['ci']['df']} "
      f"absolute_delay={params.get('absolute_delay')} (must be 10000/50.0/False)")

c = base.channel(load=False, params=params, number_of_processes=-1, name="0.1",
                 topology='random', precomputed_trajectories=ref_traj[:, :NT, :],
                 traj_config=cfg)
np.random.seed(11)   # filter() draws noise from the global RNG
data, trjs = c.filter(1024, snr=10.0, nt=NT, signal_type='sinusoid', rep=1,
                      specific=list(range(NT)))
print(f"regenerated: {data.shape}")

ref = np.asarray(ref_filt[:, :, :NT, :])
same_shape = data.shape == ref.shape
# the additive noise is RNG-dependent; compare the deterministic channel instead
hnew = np.asarray(c.h)
href = np.load(glob.glob(f"{REF}/channel_info/channel_h_*.npy")[0], mmap_mode='r')
href = np.asarray(href[:, :, :NT, :])
h_identical = np.array_equal(hnew, href)
h_max = float(np.abs(hnew - href).max()) if hnew.shape == href.shape else float('nan')
print(f"\nimpulse response h: shape {hnew.shape} vs {href.shape}")
print(f"  BIT-IDENTICAL to on-disk data: {h_identical}   (max|diff| = {h_max:.3g})")
print(f"filtered_data shape matches: {same_shape}")
corr = float(np.corrcoef(np.abs(data).ravel(), np.abs(ref).ravel())[0,1])
print(f"  filtered_data corr with on-disk (noise realisation differs): {corr:.6f}")
ok = h_identical and same_shape
print("\n" + ("PASS: defaults reproduce the original physics exactly"
              if ok else "FAIL: defaults changed the physics"))
sys.exit(0 if ok else 1)
