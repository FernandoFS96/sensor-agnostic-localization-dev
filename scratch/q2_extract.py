"""Stage 1: load N geoms once, extract compact + full features, cache to npz.
Rich per-sensor features preserve delay/phase, not just energy."""
import os,sys,glob,time,argparse
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations
ap=argparse.ArgumentParser()
ap.add_argument("--variant",default="hermite_distinct"); ap.add_argument("--theta",default="0.1")
ap.add_argument("--n_geoms",type=int,default=45); ap.add_argument("--pts",type=int,default=100)
a=ap.parse_args(); rng=np.random.default_rng(0)
root=f"data/random_task/{a.variant}"; sets=sorted(glob.glob(f"{root}/position_set_*"))[:a.n_geoms]
RMS=[];RICH=[];FULL=[];SRC=[];GID=[];CENT=[];APER=[]
t0=time.perf_counter()
for gi,sd in enumerate(sets):
    f=np.load(f"{sd}/channel_option_{a.theta}/random/filtered_data/filtered_data.npy")
    pos=np.load(glob.glob(f"{sd}/channel_option_{a.theta}/random/channel_info/sensor_positions_*.npy")[0])
    tr=np.load(f"{sd}/channel_option_{a.theta}/random/trajectory/trajectories.npy")
    tau,ppt,n_traj,ns=f.shape; xy=pos[:2].T; c=xy.mean(0)
    ap_g=max(np.linalg.norm(xy[i]-xy[j]) for i,j in combinations(range(ns),2))
    Fr=f.transpose(2,1,0,3).reshape(n_traj*ppt,tau,ns)          # (N,tau,ns)
    src=tr.transpose(1,2,0).reshape(n_traj*ppt,3)[:,:2]
    N=Fr.shape[0]; sel=rng.choice(N,size=min(a.pts,N),replace=False)
    W=Fr[sel]                                                   # (k,tau,ns)
    absW=np.abs(W); e=absW**2
    rms=np.sqrt(e.mean(1))                                      # (k,ns)
    tot=e.sum(1)+1e-12
    tidx=np.arange(tau)[None,:,None]
    mean_delay=(e*tidx).sum(1)/tot                              # (k,ns) energy centroid (delay proxy)
    peak=absW.argmax(1)                                         # (k,ns) peak delay
    maxa=absW.max(1)                                            # (k,ns)
    e1=e[:,:tau//3].sum(1); e3=e[:,2*tau//3:].sum(1); frac_late=e3/tot  # (k,ns)
    rich=np.concatenate([rms,mean_delay,peak,maxa,frac_late],axis=1)    # (k,5*ns)
    RMS.append(rms);RICH.append(rich);FULL.append(W.reshape(len(sel),tau*ns).astype(np.float32))
    SRC.append(src[sel]);GID.append(np.full(len(sel),gi));CENT.append(np.tile(c,(len(sel),1)));APER.append(np.full(len(sel),ap_g))
out=f"scratch/q2_cache_{a.variant}_{a.theta}.npz"
np.savez(out, RMS=np.concatenate(RMS),RICH=np.concatenate(RICH),FULL=np.concatenate(FULL).astype(np.float32),
         SRC=np.concatenate(SRC),GID=np.concatenate(GID),CENT=np.concatenate(CENT),APER=np.concatenate(APER),
         n_geoms=len(sets))
print(f"cached {out} in {time.perf_counter()-t0:.1f}s  rows={np.concatenate(GID).shape[0]} full_dim={tau*ns}")
