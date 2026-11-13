"""Q1: measure real simulator wall-clock. Drives the ACTUAL physics used by the
data pipeline (acoustic_data_generator.channel + filter), with the current
config knobs (df=50 -> tau=201, ppt=50). Not an estimate."""
import os, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "data"))
os.chdir(REPO)  # so any ./data side-effects resolve at repo root

import acoustic_data_generator as base

ap = argparse.ArgumentParser()
ap.add_argument("--n_traj", type=int, default=1)
ap.add_argument("--ppt", type=int, default=50)
ap.add_argument("--df", type=float, default=50.0)
ap.add_argument("--nop", type=int, default=-1)
ap.add_argument("--theta", type=float, default=0.1)
ap.add_argument("--reps", type=int, default=1)
args = ap.parse_args()

# Match production param injection (random_position_generator.override_base_params)
_orig = base.generate_params
def patched(options=None):
    p = _orig(options=options)
    p['ci']['df'] = float(args.df)
    p['ci']['fmax'] = p['ci']['fmin'] + p['ci']['B']
    p['n_traj'] = int(args.n_traj)
    p['ppt'] = int(args.ppt)
    return p
base.generate_params = patched
base.channel.save_channel_info = lambda self, name: None

cfg = base.load_traj_config()  # for trajectory method (hermite)
Lf = len(base.range_m(10000.0, 20000.0, args.df))
print(f"[cfg] method={cfg.get('method')} df={args.df} -> tau(Lf)={Lf}, n_traj={args.n_traj}, ppt={args.ppt}, nop={args.nop}, theta={args.theta}")

# Pre-generate trajectories (excluded from the physics timing below).
np.random.seed(11)
params = base.generate_params(options=args.theta)
tmp = base.channel(load=False, params=params, number_of_processes=args.nop,
                   name=str(args.theta), topology='random', traj_config=cfg)
traj = tmp.traj
print(f"[traj] shape={traj.shape}")

for r in range(args.reps):
    # Fixed layout so we only time the channel physics + filter.
    r_pos = np.zeros((3, params['n_sensors'])); r_pos[0]=np.linspace(-50,50,params['n_sensors']); r_pos[2]=params['ci']['hr0']
    orig_gsp = base.channel.generate_sensor_positions
    base.channel.generate_sensor_positions = lambda self,*a,**k: r_pos
    try:
        t0 = time.perf_counter()
        c = base.channel(load=False, params=base.generate_params(options=args.theta),
                         number_of_processes=args.nop, name=str(args.theta),
                         topology='random', precomputed_trajectories=traj, traj_config=cfg)
        t1 = time.perf_counter()   # obtain_h done
        data, trjs = c.filter(1024, snr=10.0, nt=params['n_traj'], signal_type='sinusoid',
                              rep=1, specific=list(range(params['n_traj'])))
        t2 = time.perf_counter()
    finally:
        base.channel.generate_sensor_positions = orig_gsp
    obtain = t1 - t0; filt = t2 - t1; total = t2 - t0
    per_traj = total / args.n_traj
    print(f"[rep {r}] obtain_h={obtain:.3f}s  filter={filt:.3f}s  TOTAL={total:.3f}s  "
          f"=> {per_traj:.4f}s/traj  (data {data.shape})")
