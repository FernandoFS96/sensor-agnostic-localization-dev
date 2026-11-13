"""The ENTIRE PFN task, solved classically, with NO knowledge of sensor positions:
  stage 1: from a CALIBRATION context (ranges + known source positions),
           multilaterate every sensor  -> estimated geometry
  stage 2: for held-out target points, trilaterate with the ESTIMATED geometry
This is the number the PFN has to amortize. Chance = 481 m."""
import os, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def ranges(W, tau, roll=50, df=0.5):
    env = np.abs(W); thr = 0.30*env.max(axis=-1, keepdims=True)
    return (np.argmax(env >= thr, axis=-1) - roll) * (1500.0/(tau*df))
res = {}
for NCTX in (5, 10, 20, 40):
    errs, gerrs = [], []
    for sd in sorted(glob.glob("data/random_task_bb/hermite_distinct/position_set_*"))[:8]:
        b = f"{sd}/channel_option_0.1/random"
        F = np.load(f"{b}/filtered_data/filtered_data.npy", mmap_mode='r')
        tau, ppt, ntr, ns = F.shape
        W = np.asarray(F[:, :, :12, :]).transpose(2,1,3,0)       # (12,ppt,ns,tau)
        tr = np.load(f"{b}/trajectory/trajectories.npy")[:, :12, :]
        Strue = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])[:2].T
        src = tr.transpose(1,2,0)[:,:,:2]                        # (12,ppt,2)
        R = ranges(W, tau)                                       # (12,ppt,ns)
        # context = trajectory 0 (a separate calibration transit); targets = traj 1+
        cp, cr = src[0][:NCTX], R[0][:NCTX]
        Sest = np.zeros((ns,2))
        for s in range(ns):
            A = np.c_[-2*cp, np.ones(len(cp))]
            sol,*_ = np.linalg.lstsq(A, cr[:,s]**2 - (cp**2).sum(1), rcond=None)
            Sest[s] = sol[:2]
        gerrs += list(np.linalg.norm(Sest - Strue, axis=1))
        for i in range(1, 12):                                    # held-out trajectories
            for p in range(0, ppt, 5):
                ri = R[i,p]
                A = 2*(Sest[0:1] - Sest[1:])
                bb = (ri[1:]**2 - ri[0]**2) - (np.sum(Sest[1:]**2,1) - np.sum(Sest[0]**2))
                sol,*_ = np.linalg.lstsq(A, bb, rcond=None)
                errs.append(np.linalg.norm(sol - src[i,p]))
    res[NCTX] = (np.median(gerrs), np.median(errs))
print("  ctx | geometry err | SOURCE localisation err  (no sensor positions given)")
for k,(g,e) in res.items():
    print(f"  {k:4d} | {g:9.1f} m  | {e:12.1f} m")
print(f"\n  chance = 481 m ; trained ORACLE (position GIVEN) = 484 m ; trained PFN = 473 m")
