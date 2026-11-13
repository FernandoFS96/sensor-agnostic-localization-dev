"""With 2-5 m/s tracks, pick aperture_frac (array size) and df (aliasing bound)."""
import os, sys, importlib.util
import numpy as np
from itertools import combinations
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,"data")
import acoustic_data_generator as base
import random_position_generator as rp
from omegaconf import OmegaConf
cfg = OmegaConf.create(OmegaConf.to_container(base.load_traj_config(), resolve=True))
cfg.hermite.speed_min = 2.0; cfg.hermite.speed_max = 5.0
np.random.seed(11)
p = base.generate_params(options=0.1); p['n_traj']=200; p['ppt']=50
c = base.channel.__new__(base.channel)
c.params=p; c.traj_config=cfg; c.precomputed_trajectories=None; c.topology='random'
traj = c.generate_trajectories()
xs, ys = traj[0].ravel(), traj[1].ravel()
ex = np.percentile(xs,98)-np.percentile(xs,2); ey = np.percentile(ys,98)-np.percentile(ys,2)
print(f"trajectory field extent (2-98 pct): x {ex:.0f} m, y {ey:.0f} m")
print(f"{'ap_frac':>8} {'aperture':>10} {'max range':>11} {'needed df':>10}  {'Lf@B=300':>9}")
for af in [0.10, 0.15, 0.20, 0.30, 0.50]:
    aps, mx = [], 0.0
    for s in range(12):
        pos = rp.random_sensor_positions(traj, 10, 1.0, layout_seed=1000+s,
                                         aperture_frac=af, offset_frac=0.3,
                                         scale_jitter=0.0, min_span=20.0)
        xy = pos[:2].T
        aps.append(max(np.linalg.norm(xy[i]-xy[j]) for i,j in combinations(range(10),2)))
        d = np.sqrt((xs[:,None]-xy[None,:,0])**2 + (ys[:,None]-xy[None,:,1])**2)
        mx = max(mx, d.max())
    need_df = 1500.0/(mx*1.15)     # 15% margin
    print(f"{af:8.2f} {np.mean(aps):9.0f}m {mx:10.0f}m {need_df:9.2f}Hz "
          f"{int(300/ max(need_df,1e-6))+1:9d}")
