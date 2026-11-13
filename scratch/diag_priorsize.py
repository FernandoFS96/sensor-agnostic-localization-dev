"""Trilateration needs a LAYOUT-CONDITIONAL linear solve (x = A(s)^+ b), so a
feedforward net must effectively learn a different map per layout -- which is why
it memorises. Spec 3.1 predicts exactly this: 200 fixed position-sets is far too
small a prior (TabPFN sees ~9.2M tasks). Test the prior-size ablation: does
held-out error fall as TRAINING LAYOUT COUNT grows?"""
import os, pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D="data/random_task_bb/hermite_distinct/processed_delay/geometry_split"
def load(f, stride):
    s=pickle.load(open(f"{D}/{f}","rb")); R,S,Y,G=[],[],[],[]
    for t in s[::stride]:
        X=np.asarray(t["X"]); sp=np.asarray(t["sensor_pos"])[:,:2]; n=X.shape[0]
        R.append(X[:,:10]); Y.append(np.asarray(t["y"])[:,:2])
        S.append(np.repeat(sp.reshape(1,-1),n,axis=0)); G.append(np.full(n,int(t["geometry_id"])))
    return (np.concatenate(R),np.concatenate(S),np.concatenate(Y),np.concatenate(G))
Rtr,Str,Ytr,Gtr=load("train_data.pkl",20); Rte,Ste,Yte,Gte=load("val_data.pkl",20)
def rel(S):
    P=S.reshape(len(S),10,2); return (P-P.mean(1,keepdims=True)).reshape(len(S),-1)
def cent(S): return S.reshape(len(S),10,2).mean(1)
ch=np.linalg.norm(Yte-Ytr.mean(0),axis=1).mean()
geoms=np.array(sorted(set(Gtr.tolist())))
cte=cent(Ste); Xte=np.c_[Rte, rel(Ste)/1000.0]
print(f"  layouts | train rows | HELD-OUT err   (chance {ch:.0f} m, analytic 14 m)")
for k in (10, 20, 40, 70):
    keep=set(geoms[:k].tolist()); m_=np.array([g in keep for g in Gtr])
    Xtr=np.c_[Rtr[m_], rel(Str[m_])/1000.0]; T=(Ytr[m_]-cent(Str[m_]))/1000.0
    mdl=MLPRegressor(hidden_layer_sizes=(256,256,128),max_iter=60,random_state=0).fit(Xtr,T)
    e=np.linalg.norm(mdl.predict(Xte)*1000.0+cte-Yte,axis=1).mean()
    print(f"  {k:7d} | {m_.sum():10d} | {e:9.1f} m")
