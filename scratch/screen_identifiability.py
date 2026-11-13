"""IDENTIFIABILITY SCREEN. Same probe that correctly predicted the original
failure (centroid/aperture at chance). Two tests:

 A. CLASSICAL -- from each sensor's record, take the envelope-peak delay as a
    range estimate; with the context's KNOWN source positions, multilaterate the
    sensor position by least squares. Proves whether the information is PRESENT.
 B. LEARNED   -- ridge/MLP regression of array centroid + aperture from the raw
    features on HELD-OUT geometries. Proves whether it is ACCESSIBLE.
"""
import os, sys, glob, argparse
import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--ctx", type=int, default=30, help="context pairs for multilateration")
ap.add_argument("--df", type=float, default=1.0, help="frequency resolution used at generation")
a = ap.parse_args()
sets = sorted(glob.glob(f"{a.root}/position_set_*"))
print(f"\n{'='*66}\n{a.tag}  ({len(sets)} geometries)\n{'='*66}")

FE, SRC, GID, CENT, APER, POSL = [], [], [], [], [], []
for gi, sd in enumerate(sets):
    b = f"{sd}/channel_option_0.1/random"
    F = np.load(f"{b}/filtered_data/filtered_data.npy")        # (tau,ppt,ntr,ns)
    pos = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])
    tr = np.load(f"{b}/trajectory/trajectories.npy")
    tau, ppt, ntr, ns = F.shape
    W = F.transpose(2,1,0,3)                                    # (ntr,ppt,tau,ns)
    src = tr.transpose(1,2,0)[:,:,:2]
    xy = pos[:2].T
    FE.append(W.reshape(-1, tau, ns)); SRC.append(src.reshape(-1,2))
    GID.append(np.full(ntr*ppt, gi)); POSL.append(xy)
    CENT.append(np.tile(xy.mean(0), (ntr*ppt,1)))
    APER.append(np.full(ntr*ppt, max(np.linalg.norm(xy[i]-xy[j])
                for i,j in combinations(range(ns),2))))
FE=np.concatenate(FE); SRC=np.concatenate(SRC); GID=np.concatenate(GID)
CENT=np.concatenate(CENT); APER=np.concatenate(APER)
N, tau, ns = FE.shape
print(f"features {FE.shape}  (tau={tau}, sensors={ns})   rows={N}")

# ---------- A. classical multilateration from envelope-peak delay ----------
M_PER_TAP = 1500.0 / (tau * a.df)   # c / (Lf*df): metres per delay tap
ROLL = 50                            # filter() does np.roll(h, 50): constant tap offset
rng = np.random.default_rng(0); errs = []
for gi in range(len(sets)):
    m = GID == gi
    Fg, Sg = FE[m], SRC[m]
    sel = rng.choice(len(Fg), size=min(a.ctx, len(Fg)), replace=False)
    env = np.abs(Fg[sel])                                       # (k,tau,ns)
    # FIRST arrival above a fraction of the record peak (the direct path; the
    # global max can be a surface/bottom bounce), minus the roll offset that
    # filter() introduces with np.roll(h, 50).
    thr = 0.30 * env.max(axis=1, keepdims=True)
    first = np.argmax(env >= thr, axis=1)                       # (k,ns)
    rng_est = (first - ROLL) * M_PER_TAP                        # (k,ns) metres
    for s in range(ns):
        # least-squares multilateration: |p - s_i|^2 = r_i^2  -> linearise
        P, r = Sg[sel], rng_est[:, s]
        A = np.c_[-2*P, np.ones(len(P))]
        bvec = r**2 - (P**2).sum(1)
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        errs.append(np.linalg.norm(sol[:2] - POSL[gi][s]))
errs = np.array(errs)
print(f"\nA. CLASSICAL multilateration ({a.ctx} known source positions per geometry)")
print(f"   sensor-position error: median {np.median(errs):7.1f} m | "
      f"mean {errs.mean():7.1f} m | 90th pct {np.percentile(errs,90):7.1f} m")
print(f"   -> {'GEOMETRY RECOVERABLE' if np.median(errs) < 15 else 'not recoverable this way'}")

# ---------- B. learned probe on held-out geometries ----------
ng = len(sets); perm = rng.permutation(ng); trg = set(perm[:max(1,int(ng*0.7))].tolist())
trm = np.array([g in trg for g in GID]); tem = ~trm
rmsF = np.sqrt((FE**2).mean(1))                                 # (N,ns) per-sensor RMS
env_pk = np.abs(FE).argmax(axis=1).astype(float)                # (N,ns) peak tap
def z(X): return (X - X[trm].mean(0)) / (X[trm].std(0) + 1e-9)
print(f"\nB. LEARNED probe, held-out geometries (train {len(trg)} / test {ng-len(trg)})")
for feat, nm in [(rmsF,"rms(amp)"), (env_pk,"peak-tap"), (np.c_[rmsF,env_pk],"amp+tap")]:
    X = np.c_[z(feat), z(SRC)]
    for tgt, lab in [(CENT,"centroid"), (APER[:,None],"aperture")]:
        y = tgt if tgt.shape[1]==2 else tgt.ravel()
        base = (np.tile(tgt[trm].mean(0),(tem.sum(),1)) if tgt.shape[1]==2
                else np.full((tem.sum(),1), tgt[trm].mean()))
        ch = (np.linalg.norm(base-tgt[tem],axis=1).mean() if tgt.shape[1]==2
              else np.abs(base-tgt[tem]).mean())
        out=[]
        for est,en in [(Ridge(1.0),"ridge"),
                       (MLPRegressor(hidden_layer_sizes=(128,64),max_iter=200,random_state=0),"mlp")]:
            est.fit(X[trm], y[trm]); p = est.predict(X[tem])
            p = p if tgt.shape[1]==2 else p[:,None]
            e = (np.linalg.norm(p-tgt[tem],axis=1).mean() if tgt.shape[1]==2
                 else np.abs(p-tgt[tem]).mean())
            out.append(f"{en}={e:6.1f}")
        print(f"   [{nm:9s}] {lab:9s}: " + "  ".join(out) + f"   chance={ch:6.1f} m")
