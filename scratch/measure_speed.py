"""Speed + field-size audit of the trajectory generator.
Step time is T_tot (acoustic_data_generator uses it both for Doppler,
v_t0 = (p[t+1]-p[t])/T_tot, and for the velocity histogram)."""
import os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "data")
import acoustic_data_generator as base
cfg = base.load_traj_config()
base.channel.save_channel_info = lambda self, name: None
np.random.seed(11)
p = base.generate_params(options=0.1); p['n_traj'] = 200; p['ppt'] = 50
c = base.channel.__new__(base.channel)
c.params = p; c.traj_config = cfg; c.precomputed_trajectories = None
c.topology = 'random'
traj = c.generate_trajectories()
T = float(p['ci']['T_tot'])
d = np.linalg.norm(np.diff(traj, axis=2), axis=0)     # (n_traj, ppt) step distances
v = d / T
print(f"step time T_tot = {T} s ; ppt = {p['ppt']} -> {p['ppt']*T:.0f} s observation")
print(f"SPEED  mean {v.mean():.2f}  min {v.min():.2f}  max {v.max():.2f}  "
      f"p5 {np.percentile(v,5):.2f}  p95 {np.percentile(v,95):.2f}  m/s")
frac = ((v>=2)&(v<=5)).mean()
print(f"  fraction of steps within 2-5 m/s : {frac*100:.1f}%")
L = d.sum(axis=1)
print(f"PATH   length per trajectory: mean {L.mean():.0f} m  range {L.min():.0f}-{L.max():.0f} m")
xy = traj[:2]
rad = np.linalg.norm(xy, axis=0)
print(f"FIELD  |pos| max {rad.max():.0f} m ; bbox x[{xy[0].min():.0f},{xy[0].max():.0f}] "
      f"y[{xy[1].min():.0f},{xy[1].max():.0f}]")
