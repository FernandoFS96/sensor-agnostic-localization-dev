"""Speed control: legacy must be bit-identical; enabled must hold 2-5 m/s."""
import os, sys, importlib.util
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "data")
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m
NEW=load("data/acoustic_data_generator.py","adg_new")
OLD=load("scratch/adg_BEFORE.py","adg_old")
from omegaconf import OmegaConf

def make(mod, cfg, n=200, ppt=50, seed=11):
    np.random.seed(seed)
    p = mod.generate_params(options=0.1); p['n_traj']=n; p['ppt']=ppt
    c = mod.channel.__new__(mod.channel)
    c.params=p; c.traj_config=cfg; c.precomputed_trajectories=None; c.topology='random'
    return c.generate_trajectories(), float(p['ci']['T_tot'])

base_cfg = NEW.load_traj_config()
t_old,_ = make(OLD, base_cfg)
t_new,T = make(NEW, base_cfg)
print(f"LEGACY (speed_min/max = null): bit-identical to pre-change code: "
      f"{np.array_equal(t_old, t_new)}")

cfg2 = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
cfg2.hermite.speed_min = 2.0; cfg2.hermite.speed_max = 5.0
t_s, T = make(NEW, cfg2)
d = np.linalg.norm(np.diff(t_s, axis=2), axis=0); v = d/T
print(f"\nSPEED CONTROL ON (2-5 m/s), T_tot={T}s, ppt=50 -> {50*T:.0f}s track:")
print(f"  per-step speed: mean {v.mean():.2f}  min {v.min():.2f}  max {v.max():.2f} m/s")
print(f"  fraction of steps within 2-5 m/s : {((v>=2)&(v<=5)).mean()*100:.1f}%")
pv = v.mean(axis=1)
print(f"  per-trajectory speed spread: {pv.min():.2f}-{pv.max():.2f} m/s "
      f"(within-traj std {v.std(axis=1).mean():.4f} -> constant by construction)")
L = d.sum(axis=1)
print(f"  path length: mean {L.mean():.0f} m  range {L.min():.0f}-{L.max():.0f} m")
xy=t_s[:2]; print(f"  field |pos| max {np.linalg.norm(xy,axis=0).max():.0f} m")
