"""Stage 2: probes on cached features (Ridge + MLP), held-out-geometry split."""
import sys,numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
d=np.load(sys.argv[1]); rng=np.random.default_rng(1)
RMS,RICH,FULL,SRC,GID,CENT,APER=d['RMS'],d['RICH'],d['FULL'],d['SRC'],d['GID'],d['CENT'],d['APER']
ng=int(d['n_geoms']); perm=rng.permutation(ng); tr_g=set(perm[:int(ng*0.7)].tolist())
trm=np.array([g in tr_g for g in GID]); tem=~trm
print(f"geoms={ng} (train {len(tr_g)}/test {ng-len(tr_g)})  rows train={trm.sum()} test={tem.sum()}")
print(f"centroid spread: x_std={CENT[:,0].std():.0f} y_std={CENT[:,1].std():.0f}  aperture std={APER.std():.1f}")

def z(X):
    m=X[trm].mean(0); s=X[trm].std(0)+1e-9; return (X-m)/s

def mae(pr,tgt):
    return np.linalg.norm(pr-tgt[tem],axis=1).mean() if tgt.shape[1]==2 else np.abs(pr-tgt[tem]).mean()

def geom_pooled(p,tgt):
    gg=np.unique(GID[tem]); pe=[]
    for g in gg:
        m=GID[tem]==g
        if tgt.shape[1]==2: pe.append(np.linalg.norm(p[m].mean(0)-tgt[tem][m][0]))
        else: pe.append(abs(p[m].mean(0)-tgt[tem][m][0]))
    return float(np.mean(pe))

def run(feat,name,use_mlp=False):
    X=np.concatenate([z(feat),z(SRC)],axis=1)
    for tgt,lab in [(CENT,"centroid_xy"),(APER[:,None],"aperture")]:
        y=tgt if tgt.shape[1]==2 else tgt.ravel()
        r=Ridge(alpha=1.0).fit(X[trm],y[trm]); p=r.predict(X[tem]); p=p if tgt.shape[1]==2 else p[:,None]
        base=np.tile(tgt[trm].mean(0),(tem.sum(),1)) if tgt.shape[1]==2 else np.full((tem.sum(),1),tgt[trm].mean())
        line=(f"  [{name:9s}] {lab:12s} ridge point={mae(p,tgt):6.1f}  "
              f"geom-pooled={geom_pooled(p,tgt):6.1f}  chance={mae(base,tgt):6.1f}")
        if use_mlp:
            mm=MLPRegressor(hidden_layer_sizes=(128,64),max_iter=200,random_state=0).fit(X[trm],y[trm])
            pm=mm.predict(X[tem]); pm=pm if tgt.shape[1]==2 else pm[:,None]
            line+=f"  | mlp point={mae(pm,tgt):6.1f} geom-pooled={geom_pooled(pm,tgt):6.1f}"
        print(line)

print("\n(B) held-out-geometry descriptor regression (input=[features, source_xy]):")
run(RMS,"rms10")
run(RICH,"rich50",use_mlp=True)
run(FULL,"full2010",use_mlp=True)
print("\n(A) geometry-identity classification (kNN, within-geom point split):")
for feat,nm in [(RMS,"rms10"),(RICH,"rich50"),(FULL,"full2010")]:
    Z=z(feat); idx=rng.permutation(len(GID)); h=len(idx)//2
    k=KNeighborsClassifier(5).fit(Z[idx[:h]],GID[idx[:h]]); acc=(k.predict(Z[idx[h:]])==GID[idx[h:]]).mean()
    print(f"  [{nm:9s}] acc={acc*100:.1f}%  chance={100/ng:.2f}%")
