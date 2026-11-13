"""Controls: does ACOUSTICS carry centroid/aperture info beyond source position?
Compare source-only vs acoustics-only vs both, on held-out geometries."""
import sys,numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
d=np.load(sys.argv[1]); rng=np.random.default_rng(1)
RMS,FULL,SRC,GID,CENT,APER=d['RMS'],d['FULL'],d['SRC'],d['GID'],d['CENT'],d['APER']
ng=int(d['n_geoms']); perm=rng.permutation(ng); tr_g=set(perm[:int(ng*0.7)].tolist())
trm=np.array([g in tr_g for g in GID]); tem=~trm
def z(X): m=X[trm].mean(0); s=X[trm].std(0)+1e-9; return (X-m)/s
def geom_pooled(p,tgt):
    gg=np.unique(GID[tem]); pe=[]
    for g in gg:
        m=GID[tem]==g
        pe.append(np.linalg.norm(p[m].mean(0)-tgt[tem][m][0]) if tgt.shape[1]==2 else abs(p[m].mean()-tgt[tem][m].mean()))
    return float(np.mean(pe))
def fit(feat,tgt,mlp=False):
    X=feat; y=tgt if tgt.shape[1]==2 else tgt.ravel()
    est=MLPRegressor(hidden_layer_sizes=(128,64),max_iter=300,random_state=0) if mlp else Ridge(1.0)
    est.fit(X[trm],y[trm]); p=est.predict(X[tem]); p=p if tgt.shape[1]==2 else p[:,None]
    return geom_pooled(p,tgt)
SRCz=z(SRC); RMSz=z(RMS); FULLz=z(FULL)
both=np.concatenate([RMSz,SRCz],1); bothF=np.concatenate([FULLz,SRCz],1)
for tgt,lab in [(CENT,"centroid_xy"),(APER[:,None],"aperture")]:
    base=np.tile(tgt[trm].mean(0),(tem.sum(),1)) if tgt.shape[1]==2 else np.full((tem.sum(),1),tgt[trm].mean())
    ch=geom_pooled(base,tgt)
    print(f"\n{lab}: chance(predict-mean) geom-pooled MAE = {ch:.1f}m")
    print(f"  source-xy ONLY      : ridge {fit(SRCz,tgt):.1f}   mlp {fit(SRCz,tgt,1):.1f}")
    print(f"  rms10   ONLY        : ridge {fit(RMSz,tgt):.1f}   mlp {fit(RMSz,tgt,1):.1f}")
    print(f"  full2010 ONLY       : ridge {fit(FULLz,tgt):.1f}   mlp {fit(FULLz,tgt,1):.1f}")
    print(f"  rms10 + source      : ridge {fit(both,tgt):.1f}   mlp {fit(both,tgt,1):.1f}")
    print(f"  full2010 + source   : ridge {fit(bothF,tgt):.1f}   mlp {fit(bothF,tgt,1):.1f}")
