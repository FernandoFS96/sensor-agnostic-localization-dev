"""Chance baseline on the broadband dataset (predict the global-mean source
position), plus a quick speed/geometry audit of what was actually generated."""
import os, glob
import numpy as np
from itertools import combinations
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = "data/random_task_bb/hermite_distinct"
sets = sorted(glob.glob(f"{R}/position_set_*"))
Y, aps, mx, spd = [], [], 0.0, []
for sd in sets[:40]:
    for th in ("0.1","0.2"):
        t = f"{sd}/channel_option_{th}/random/trajectory/trajectories.npy"
        if not os.path.exists(t): continue
        tr = np.load(t); s = tr.transpose(1,2,0)[:,:,:2].reshape(-1,2); Y.append(s)
        p = np.load(glob.glob(f"{sd}/channel_option_{th}/random/channel_info/sensor_positions_*.npy")[0])
        xy = p[:2].T
        aps.append(max(np.linalg.norm(xy[i]-xy[j]) for i,j in combinations(range(10),2)))
        mx = max(mx, np.linalg.norm(s[:,None,:]-xy[None,:,:],axis=-1).max())
        full = np.load(glob.glob(f"{sd}/channel_option_{th}/random/channel_info/trajs_*.npy")[0])
        spd.append((np.linalg.norm(np.diff(full,axis=2),axis=0)/6.0).ravel())
Y = np.concatenate(Y); spd = np.concatenate(spd)
gm = Y.mean(0)
print(f"CHANCE (predict global-mean source position) = {np.linalg.norm(Y-gm,axis=1).mean():.1f} m")
print(f"  source field: x[{Y[:,0].min():.0f},{Y[:,0].max():.0f}] y[{Y[:,1].min():.0f},{Y[:,1].max():.0f}]")
print(f"SPEED  mean {spd.mean():.2f}  min {spd.min():.2f}  max {spd.max():.2f} m/s | "
      f"in 2-5 band {((spd>=2)&(spd<=5)).mean()*100:.0f}%")
print(f"ARRAY  aperture mean {np.mean(aps):.0f} m (range {min(aps):.0f}-{max(aps):.0f})")
print(f"RANGE  max source-sensor {mx:.0f} m  vs unambiguous c/df=3000 m -> "
      f"{'OK' if mx<3000 else 'ALIASING'}")
