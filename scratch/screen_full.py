"""Identifiability screen at FULL geometry count (the 8-geometry learned probe
was underpowered). Trajectory-subsampled to keep memory sane."""
import os, glob, argparse
import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/random_task_bb/hermite_distinct")
ap.add_argument("--geoms", type=int, default=40)
ap.add_argument("--trajs", type=int, default=10)
ap.add_argument("--df", type=float, default=0.5)
ap.add_argument("--ctx", type=int, default=30)
a = ap.parse_args()
rng = np.random.default_rng(0)
sets = sorted(glob.glob(f"{a.root}/position_set_*"))[:a.geoms]
FE, SRC, GID, CENT, APER, POSL = [], [], [], [], [], []
for gi, sd in enumerate(sets):
    b = f"{sd}/channel_option_0.1/random"
    F = np.load(f"{b}/filtered_data/filtered_data.npy", mmap_mode='r')
    tau, ppt, ntr, ns = F.shape
    sel = rng.choice(ntr, size=min(a.trajs, ntr), replace=False)
    W = np.asarray(F[:, :, sel, :]).transpose(2,1,0,3).astype(np.float32)  # (k,ppt,tau,ns)
    tr = np.load(f"{b}/trajectory/trajectories.npy")[:, sel, :]
    pos = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0]); xy = pos[:2].T
    FE.append(W.reshape(-1, tau, ns)); SRC.append(tr.transpose(1,2,0)[:,:,:2].reshape(-1,2))
    n = len(sel)*ppt
    GID.append(np.full(n, gi)); POSL.append(xy)
    CENT.append(np.tile(xy.mean(0),(n,1)))
    APER.append(np.full(n, max(np.linalg.norm(xy[i]-xy[j]) for i,j in combinations(range(ns),2))))
FE=np.concatenate(FE); SRC=np.concatenate(SRC); GID=np.concatenate(GID)
CENT=np.concatenate(CENT); APER=np.concatenate(APER)
N, tau, ns = FE.shape
print(f"{len(sets)} geometries, {a.trajs} traj each -> {N} rows, tau={tau}")

MPT = 1500.0/(tau*a.df); ROLL = 50
errs=[]
for gi in range(len(sets)):
    m = GID==gi; Fg,Sg = FE[m],SRC[m]
    sel = rng.choice(len(Fg), size=min(a.ctx,len(Fg)), replace=False)
    env = np.abs(Fg[sel]); thr = 0.30*env.max(axis=1,keepdims=True)
    r_est = (np.argmax(env>=thr,axis=1)-ROLL)*MPT
    for s in range(ns):
        P,r = Sg[sel], r_est[:,s]
        sol,*_ = np.linalg.lstsq(np.c_[-2*P,np.ones(len(P))], r**2-(P**2).sum(1), rcond=None)
        errs.append(np.linalg.norm(sol[:2]-POSL[gi][s]))
print(f"A. CLASSICAL multilateration ({a.ctx} ctx pts): median "
      f"{np.median(errs):.1f} m | 90th pct {np.percentile(errs,90):.1f} m")

ng=len(sets); perm=rng.permutation(ng); trg=set(perm[:int(ng*0.7)].tolist())
trm=np.array([g in trg for g in GID]); tem=~trm
rms=np.sqrt((FE**2).mean(1)); pk=np.abs(FE).argmax(axis=1).astype(np.float32)
def z(X): return (X-X[trm].mean(0))/(X[trm].std(0)+1e-9)
print(f"B. LEARNED probe, held-out geometries (train {len(trg)} / test {ng-len(trg)})")
for feat,nm in [(rms,"rms(amp)"),(pk,"peak-tap"),(np.c_[rms,pk],"amp+tap")]:
    X=np.c_[z(feat), z(SRC)]
    for tgt,lab in [(CENT,"centroid"),(APER[:,None],"aperture")]:
        y = tgt if tgt.shape[1]==2 else tgt.ravel()
        base=(np.tile(tgt[trm].mean(0),(tem.sum(),1)) if tgt.shape[1]==2
              else np.full((tem.sum(),1),tgt[trm].mean()))
        ch=(np.linalg.norm(base-tgt[tem],axis=1).mean() if tgt.shape[1]==2
            else np.abs(base-tgt[tem]).mean())
        out=[]
        for est,en in [(Ridge(1.0),"ridge"),
                       (MLPRegressor(hidden_layer_sizes=(128,64),max_iter=150,random_state=0),"mlp")]:
            est.fit(X[trm],y[trm]); p=est.predict(X[tem]); p=p if tgt.shape[1]==2 else p[:,None]
            e=(np.linalg.norm(p-tgt[tem],axis=1).mean() if tgt.shape[1]==2
               else np.abs(p-tgt[tem]).mean())
            out.append(f"{en}={e:6.1f}")
        print(f"   [{nm:9s}] {lab:9s}: "+"  ".join(out)+f"   chance={ch:6.1f} m")
