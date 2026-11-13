"""SCREEN B, step 1 -- generate data where the ENVIRONMENT is the latent.

Fixed, known array. What varies between "environments" is the ocean:
  c   sound speed        [m/s]   -- sets the delay scale
  h0  water depth        [m]     -- sets the multipath bounce geometry
  c2  bottom sound speed [m/s]   -- sets the bottom reflection coefficient
All three are genuinely unknown at deployment and change day to day, unlike the
array geometry, which you can survey once. This is the precondition for the
"environment as latent" paradigm; the screen that follows asks whether they are
RECOVERABLE from a labelled context set -- the same question that killed the
geometry version, asked before committing this time.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "data")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "data")
import acoustic_data_generator as base

N_ENV, N_TRAJ, PPT = 40, 12, 50
B, DF, SIGNAL_N = 300.0, 0.5, 2048
OUT = "scratch/screenB_env.npz"

cfg = base.load_traj_config()
base.channel.save_channel_info = lambda self, name: None
rng = np.random.default_rng(0)

# ONE fixed array for every environment (a 450 m line, the agreed operating point)
S = 10
r_pos = np.zeros((3, S)); r_pos[0] = np.linspace(-225, 225, S); r_pos[2] = 1.0
orig_gsp = base.channel.generate_sensor_positions
base.channel.generate_sensor_positions = lambda self, *a, **k: r_pos

feats, srcs, lats = [], [], []
t0 = time.time()
for e in range(N_ENV):
    c  = rng.uniform(1450.0, 1550.0)
    h0 = rng.uniform(30.0, 80.0)
    c2 = rng.uniform(1150.0, 1350.0)
    np.random.seed(1000 + e)
    p = base.generate_params(options=0.1)
    p['ci'].update(B=B, df=DF, c=c, h0=h0, c2=c2, ht0=h0)   # ht0 tied to depth
    p['ci']['fmax'] = p['ci']['fmin'] + B
    p['n_traj'] = N_TRAJ; p['ppt'] = PPT
    p['absolute_delay'] = True; p['matched_filter'] = True
    ch = base.channel(load=False, params=p, number_of_processes=-1, name="env",
                      topology='random', traj_config=cfg)
    data, trj = ch.filter(SIGNAL_N, snr=10.0, nt=N_TRAJ, signal_type='chirp',
                          rep=1, specific=list(range(N_TRAJ)))
    tau = data.shape[0]
    feats.append(data.transpose(2, 1, 0, 3).reshape(N_TRAJ * PPT, tau, S).astype(np.float32))
    srcs.append(trj.transpose(1, 2, 0)[:, :, :2].reshape(N_TRAJ * PPT, 2).astype(np.float32))
    lats.append(np.tile([c, h0, c2], (N_TRAJ * PPT, 1)))
    if e % 10 == 0:
        print(f"  env {e}/{N_ENV}  c={c:.0f} h0={h0:.0f} c2={c2:.0f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
base.channel.generate_sensor_positions = orig_gsp
env_id = np.repeat(np.arange(N_ENV), N_TRAJ * PPT)
np.savez_compressed(OUT, feats=np.concatenate(feats), src=np.concatenate(srcs),
                    lat=np.concatenate(lats), env=env_id, sensors=r_pos[:2].T,
                    tau=tau, df=DF)
print(f"  wrote {OUT}  feats={np.concatenate(feats).shape}  in {time.time()-t0:.0f}s")
