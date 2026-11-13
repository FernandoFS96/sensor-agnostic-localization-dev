"""DECISIVE CONTROL: is the BLIND model's context curve driven by ACOUSTICS
(geometry inference) or by TRAJECTORY MEMORIZATION (shortcut)?

Scramble the acoustic features X across samples while keeping (context_y, target_y)
intact. If MAE stays low -> the model ignores acoustics and just extrapolates the
revealed trajectory prefix (shortcut). If MAE collapses to ~chance -> acoustics
carry the signal and in-context geometry inference is real."""
import os,sys,json,argparse,numpy as np, torch as t
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec=importlib.util.spec_from_file_location("evalmod","scripts/eval/eval_np_geometry.py")
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--ctx",default="10,40"); ap.add_argument("--per_geom",type=int,default=25)
ap.add_argument("--tag",default=""); a=ap.parse_args()
dev=t.device("cuda" if t.cuda.is_available() else "cpu")
ck=t.load(a.ckpt,map_location=dev); cfg=ck.get("config",{})
name=ck.get("model_name","cnp"); conv=ck.get("convention","split")
data_dir=cfg.get("data_dir")
if not os.path.isabs(data_dir): data_dir=os.path.normpath(os.path.join(".",data_dir))
sc=cfg.get("spatial",None)
model,conv2=E.build_model(name,cfg.get("num_hidden",128),ck.get("feat_dim"),ck.get("out_dim",3),dropout=cfg.get("dropout",0.1),spatial_cfg=sc)
conv=conv or conv2; model.to(dev); model.load_state_dict(ck["model"]); model.eval()
ym=ck.get("y_mean"); ys=ck.get("y_std")
csm=cfg.get("ctx_sample_mode","first"); exc=bool(cfg.get("exclude_ctx_from_target",True))
ds=E.TrajectoryDataset(os.path.join(data_dir,"val_data.pkl"))
class Sub:
    """mode: 'real' = true X ; 'scramble' = X taken from a DIFFERENT random sample
       (y untouched, so the revealed context prefix is still the true trajectory)"""
    def __init__(s,ds,per_geom,mode,seed=0):
        r=np.random.default_rng(seed); by={}
        for i,smp in enumerate(ds.samples): by.setdefault(int(smp.get("geometry_id",-1)),[]).append(i)
        s.keep=[]
        for g,idxs in by.items():
            idxs=np.array(idxs); s.keep+=list(idxs if len(idxs)<=per_geom else r.choice(idxs,per_geom,replace=False))
        s.ds=ds; s.mode=mode
        s.alt=r.permutation(len(ds.samples))   # donor index for scrambled acoustics
    def __len__(s): return len(s.keep)
    def __getitem__(s,i):
        j=s.keep[i]; X,y,g,th,spv=s.ds[j]
        if s.mode=="scramble":
            Xd,_,_,_,_=s.ds[int(s.alt[j])]     # acoustics from an unrelated sample
            X=Xd
        return X,y,g,th,spv
print(f"[{a.tag}] use_position={sc.get('use_position') if sc else None} epoch={ck.get('epoch')}",flush=True)
for c in [int(x) for x in a.ctx.split(",")]:
    row=[]
    for mode in ("real","scramble"):
        sub=Sub(ds,a.per_geom,mode)
        gm=E.per_geometry_mae(model,conv,sub,dev,c,1,y_mean=ym,y_std=ys,ctx_sample_mode=csm,exclude_ctx_from_target=exc)
        row.append(f"{mode}={float(np.mean(list(gm.values()))):7.2f}")
    print(f"  ctx={c:3d}:  "+"   ".join(row),flush=True)
