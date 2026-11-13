"""Can the SOURCE be localised from these features given TRUE sensor positions?
(the ORACLE's exact task). Analytic trilateration from per-sensor peak delay."""
import os, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
errs=[]
for sd in sorted(glob.glob("data/random_task_bb/hermite_distinct/position_set_*"))[:6]:
    b=f"{sd}/channel_option_0.1/random"
    F=np.load(f"{b}/filtered_data/filtered_data.npy", mmap_mode='r')
    tau,ppt,ntr,ns=F.shape
    W=np.asarray(F[:,:,:15,:]).transpose(2,1,3,0)          # (15,ppt,ns,tau)
    tr=np.load(f"{b}/trajectory/trajectories.npy")[:,:15,:]
    pos=np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])
    S=pos[:2].T                                            # (ns,2) TRUE sensor pos
    src=tr.transpose(1,2,0)[:,:,:2]                        # (15,ppt,2)
    MPT=1500.0/(tau*0.5); ROLL=50
    env=np.abs(W); thr=0.30*env.max(axis=-1,keepdims=True)
    first=np.argmax(env>=thr,axis=-1)
    r=(first-ROLL)*MPT                                     # (15,ppt,ns) ranges
    for i in range(r.shape[0]):
        for p in range(r.shape[1]):
            ri=r[i,p]
            # |x - S_j|^2 = r_j^2 ; subtract eq 0 -> linear in x
            A=2*(S[0:1]-S[1:]); bb=(ri[1:]**2-ri[0]**2)-(np.sum(S[1:]**2,1)-np.sum(S[0]**2))
            sol,*_=np.linalg.lstsq(A,bb,rcond=None)
            errs.append(np.linalg.norm(sol-src[i,p]))
errs=np.array(errs)
print(f"SOURCE localisation from features + TRUE sensor positions (analytic):")
print(f"  median {np.median(errs):.1f} m | mean {errs.mean():.1f} m | "
      f"90th pct {np.percentile(errs,90):.1f} m   (n={len(errs)}, chance=481 m)")
print(f"  -> the ORACLE task IS solvable from these features"
      if np.median(errs)<30 else "  -> NOT solvable this way")
