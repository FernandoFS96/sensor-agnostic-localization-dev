"""Do the STORED delay features carry range? Recompute the analytic baselines
from exactly what the trainer will see."""
import os, sys, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,"data"); import process_data as P
errs_src, errs_geo, corrs = [], [], []
for sd in sorted(glob.glob("data/random_task_bb/hermite_distinct/position_set_*"))[:6]:
    b=f"{sd}/channel_option_0.1/random"
    F=np.load(f"{b}/filtered_data/filtered_data.npy", mmap_mode='r')
    tau,ppt,ntr,ns=F.shape
    sub=np.asarray(F[:,:,:12,:])
    X=P.delay_features(sub, 1500.0/(tau*0.5), roll=50, thresh=0.30)   # (12,ppt,40)
    tr=np.load(f"{b}/trajectory/trajectories.npy")[:,:12,:]
    S=np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])[:2].T
    src=tr.transpose(1,2,0)[:,:,:2]
    r=X[...,:ns]*1000.0                       # feature 0 = range_first, back to metres
    d=np.linalg.norm(src[:,:,None,:]-S[None,None,:,:],axis=-1)
    corrs.append(np.corrcoef(r.ravel(), d.ravel())[0,1])
    # stage 1: geometry from a 40-point spread calibration transit (traj 0)
    sel=np.linspace(0,ppt-1,40).astype(int); cp,cr=src[0][sel],r[0][sel]
    Sest=np.zeros((ns,2))
    for s in range(ns):
        sol,*_=np.linalg.lstsq(np.c_[-2*cp,np.ones(len(cp))], cr[:,s]**2-(cp**2).sum(1), rcond=None)
        Sest[s]=sol[:2]
    errs_geo += list(np.linalg.norm(Sest-S,axis=1))
    # stage 2: localise held-out trajectories with the ESTIMATED geometry
    for i in range(1,12):
        for p in range(0,ppt,5):
            ri=r[i,p]
            A=2*(Sest[0:1]-Sest[1:]); bb=(ri[1:]**2-ri[0]**2)-(np.sum(Sest[1:]**2,1)-np.sum(Sest[0]**2))
            sol,*_=np.linalg.lstsq(A,bb,rcond=None)
            errs_src.append(np.linalg.norm(sol-src[i,p]))
print(f"  corr(stored range feature, TRUE range) = {np.mean(corrs):+.4f}")
print(f"  geometry from 40 spread ctx pts        : median {np.median(errs_geo):.1f} m")
print(f"  SOURCE localisation, geometry inferred : median {np.median(errs_src):.1f} m   (chance 481)")
