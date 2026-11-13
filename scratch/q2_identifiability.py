"""Q2: is array geometry identifiable from the acoustic features?
Two probes, both trajectory-controlled and using held-out GEOMETRIES:
 (A) geometry classification at matched-ish source positions (decodability).
 (B) regress permutation-invariant geometry descriptors (array centroid xy,
     aperture) from acoustic features -> the quantity the PFN must recover.
Features: per-sensor RMS energy (10-d, a physical range proxy) AND full 2010-d.
"""
import os, sys, glob, time, argparse
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
os.chdir(REPO)
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier

ap=argparse.ArgumentParser()
ap.add_argument("--variant",default="hermite_distinct")
ap.add_argument("--theta",default="0.1")
ap.add_argument("--n_geoms",type=int,default=60)
ap.add_argument("--pts_per_geom",type=int,default=60)
ap.add_argument("--full",action="store_true",help="also test full 2010-d features")
args=ap.parse_args()
rng=np.random.default_rng(0)

root=f"data/random_task/{args.variant}"
sets=sorted(glob.glob(f"{root}/position_set_*"))
sets=sets[:args.n_geoms]
print(f"variant={args.variant} theta={args.theta}  using {len(sets)} geometries, "
      f"{args.pts_per_geom} pts/geom")

def load(sd):
    f=np.load(f"{sd}/channel_option_{args.theta}/random/filtered_data/filtered_data.npy") # (tau,ppt,n_traj,ns)
    pos=np.load(glob.glob(f"{sd}/channel_option_{args.theta}/random/channel_info/sensor_positions_*.npy")[0]) # (3,ns)
    tr=np.load(f"{sd}/channel_option_{args.theta}/random/trajectory/trajectories.npy") # (3,n_traj,ppt)
    return f,pos,tr

# Build per-sample rows: per-sensor RMS (10-d), full (2010-d), source xy (2), + geom descriptors
rms_rows=[]; full_rows=[]; src_rows=[]; gid=[]; cent=[]; aper=[]
t0=time.perf_counter()
for gi,sd in enumerate(sets):
    f,pos,tr=load(sd)
    tau,ppt,n_traj,ns=f.shape
    xy=pos[:2].T                      # (ns,2)
    c=xy.mean(0)                      # centroid
    ap_g=max(np.linalg.norm(xy[a]-xy[b]) for a,b in combinations(range(ns),2))
    # flatten (traj,pt); features tau-major/sensor-minor -> (N, tau, ns)
    Fr=f.transpose(2,1,0,3).reshape(n_traj*ppt, tau, ns)   # (N,tau,ns)
    src=tr.transpose(1,2,0).reshape(n_traj*ppt,3)[:,:2]    # (N,2) source xy
    N=Fr.shape[0]
    sel=rng.choice(N, size=min(args.pts_per_geom,N), replace=False)
    rms=np.sqrt((Fr[sel]**2).mean(axis=1))                 # (k,ns) per-sensor RMS
    rms_rows.append(rms); src_rows.append(src[sel])
    if args.full: full_rows.append(Fr[sel].reshape(len(sel),tau*ns))
    gid.append(np.full(len(sel),gi)); cent.append(np.tile(c,(len(sel),1))); aper.append(np.full(len(sel),ap_g))
print(f"loaded in {time.perf_counter()-t0:.1f}s")

RMS=np.concatenate(rms_rows); SRC=np.concatenate(src_rows); GID=np.concatenate(gid)
CENT=np.concatenate(cent); APER=np.concatenate(aper)
print(f"rows={len(GID)}  rms_dim={RMS.shape[1]}  centroid range x[{CENT[:,0].min():.0f},{CENT[:,0].max():.0f}] "
      f"y[{CENT[:,1].min():.0f},{CENT[:,1].max():.0f}]  aperture[{APER.min():.0f},{APER.max():.0f}]")

# ---- split by GEOMETRY (held-out), the PFN-relevant generalization ----
ng=len(sets); perm=rng.permutation(ng); n_tr=int(ng*0.7)
train_g=set(perm[:n_tr].tolist()); 
tr_mask=np.array([g in train_g for g in GID]); te_mask=~tr_mask
print(f"train geoms={len(train_g)} test geoms={ng-len(train_g)} | train rows={tr_mask.sum()} test rows={te_mask.sum()}")

def stand_fit(X): 
    m=X.mean(0); s=X.std(0)+1e-9; return m,s
def zz(X,m,s): return (X-m)/s

def probe(Xfeat, name):
    m,s=stand_fit(Xfeat[tr_mask]); Xz=zz(Xfeat,m,s)
    # input = [features, source_xy]  (source known in the PFN context)
    X=np.concatenate([Xz, SRC],axis=1)
    for target,label,unit in [(CENT,"centroid_xy","m"),(APER[:,None],"aperture","m")]:
        reg=Ridge(alpha=1.0).fit(X[tr_mask],target[tr_mask])
        pred=reg.predict(X[te_mask])
        # per-geometry aggregate (context = whole set): mean prediction per test geom
        err=np.linalg.norm(pred-target[te_mask],axis=1) if target.shape[1]==2 else np.abs(pred-target[te_mask]).ravel()
        # baseline: predict train mean
        base=target[tr_mask].mean(0)
        berr=np.linalg.norm(target[te_mask]-base,axis=1) if target.shape[1]==2 else np.abs(target[te_mask]-base).ravel()
        # aggregate per-geometry
        geoms=np.unique(GID[te_mask]); ge=[]; gb=[]
        for g in geoms:
            gm=GID[te_mask]==g
            gp=pred[gm].mean(0); gt=target[te_mask][gm][0]
            ge.append(np.linalg.norm(gp-gt) if target.shape[1]==2 else abs(gp-gt).item())
        print(f"  [{name:8s}] {label:12s}: per-point MAE={err.mean():.1f}{unit}  "
              f"per-geom(context-pooled) MAE={np.mean(ge):.1f}{unit}  | chance(mean)={berr.mean():.1f}{unit}")

print("\n(B) held-out-geometry descriptor regression (Ridge; input=[features, source_xy]):")
probe(RMS,"rms10")
if args.full: probe(np.concatenate(full_rows),"full2010")

# ---- (A) geometry classification at matched positions (kNN on RMS) ----
# train/test split by POINTS within all geoms (decodability of identity)
print("\n(A) geometry classification (kNN, per-sensor RMS, points split within geoms):")
mrms,srms=stand_fit(RMS); Rz=zz(RMS,mrms,srms)
idx=rng.permutation(len(GID)); h=len(idx)//2
knn=KNeighborsClassifier(n_neighbors=5).fit(Rz[idx[:h]],GID[idx[:h]])
acc=(knn.predict(Rz[idx[h:]])==GID[idx[h:]]).mean()
print(f"  accuracy={acc*100:.1f}%   chance={100.0/ng:.2f}%   (n_geoms={ng})")
