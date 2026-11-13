"""Trilateration is translation-EQUIVARIANT: given ranges and the array's SHAPE,
the source position relative to the array is a fixed function, independent of
where the array sits. The current model works in ABSOLUTE coordinates, so every
new layout is a new region of a 3 km field and 70 training layouts are far too
few -> it memorises. Test: express everything in the ARRAY frame."""
import os, pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D="data/random_task_bb/hermite_distinct/processed_delay/geometry_split"
def load(f, stride):
    s=pickle.load(open(f"{D}/{f}","rb")); R,S,Y=[],[],[]
    for t in s[::stride]:
        X=np.asarray(t["X"]); sp=np.asarray(t["sensor_pos"])[:,:2]
        R.append(X[:,:10]); Y.append(np.asarray(t["y"])[:,:2])
        S.append(np.repeat(sp.reshape(1,-1),X.shape[0],axis=0))
    return np.concatenate(R), np.concatenate(S), np.concatenate(Y)
Rtr,Str,Ytr=load("train_data.pkl",40); Rte,Ste,Yte=load("val_data.pkl",40)
ch=np.linalg.norm(Yte-Ytr.mean(0),axis=1).mean()
def cent(S): return S.reshape(len(S),10,2).mean(1)
def rel(S):  # centroid-relative sensor offsets (the array SHAPE)
    P=S.reshape(len(S),10,2); return (P-P.mean(1,keepdims=True)).reshape(len(S),-1)

def fit(Xtr,Ttr,Xte,Tte,off_tr,off_te):
    m=MLPRegressor(hidden_layer_sizes=(256,256,128),max_iter=60,random_state=0)
    m.fit(Xtr,Ttr/1000.0)
    e=np.linalg.norm(m.predict(Xte)*1000.0+off_te-Yte,axis=1)
    t=np.linalg.norm(m.predict(Xtr)*1000.0+off_tr-Ytr,axis=1)
    return t.mean(), e.mean()

z=np.zeros((len(Yte),2)); ztr=np.zeros((len(Ytr),2))
a=fit(np.c_[Rtr,Str/1000.0],Ytr,np.c_[Rte,Ste/1000.0],Yte,ztr,z)
print(f"  ABSOLUTE frame (current)          train {a[0]:6.1f} m | HELD-OUT {a[1]:6.1f} m")
ctr,cte=cent(Str),cent(Ste)
b=fit(np.c_[Rtr,rel(Str)/1000.0],Ytr-ctr,np.c_[Rte,rel(Ste)/1000.0],Yte-cte,ctr,cte)
print(f"  ARRAY frame (centroid-relative)   train {b[0]:6.1f} m | HELD-OUT {b[1]:6.1f} m")
print(f"  chance                                          | {ch:16.1f} m")
print(f"  analytic trilateration                          | {14.0:16.1f} m")
