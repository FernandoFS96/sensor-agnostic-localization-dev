"""Does the context set carry task identity? MAE vs context size for a given
checkpoint, matched protocol. Also computes the chance baseline (predict the
global mean source position) so gap-closure can be stated rigorously."""
import os,sys,json,argparse,numpy as np, torch as t
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec=importlib.util.spec_from_file_location("evalmod","scripts/eval/eval_np_geometry.py")
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--ctx",default="1,2,5,10,20,40"); ap.add_argument("--per_geom",type=int,default=25)
ap.add_argument("--pools",default="val"); ap.add_argument("--tag",default="")
a=ap.parse_args()
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
labels={}; sp=os.path.join(data_dir,"splits.json")
if os.path.exists(sp): labels={int(k):v for k,v in json.load(open(sp)).get("labels",{}).items()}
class Sub:
    def __init__(s,ds,per_geom,seed=0):
        r=np.random.default_rng(seed); by={}
        for i,smp in enumerate(ds.samples): by.setdefault(int(smp.get("geometry_id",-1)),[]).append(i)
        s.keep=[]
        for g,idxs in by.items():
            idxs=np.array(idxs); s.keep+=list(idxs if len(idxs)<=per_geom else r.choice(idxs,per_geom,replace=False))
        s.ds=ds
    def __len__(s): return len(s.keep)
    def __getitem__(s,i): return s.ds[s.keep[i]]
pools={nm:E.TrajectoryDataset(os.path.join(data_dir,f"{nm}_data.pkl")) for nm in a.pools.split(",")}
print(f"[{a.tag}] ckpt epoch={ck.get('epoch')} use_position={sc.get('use_position') if sc else None}",flush=True)
# chance baseline: predict global-mean source position (train pool stats)
tr=E.TrajectoryDataset(os.path.join(data_dir,"train_data.pkl"))
allY=np.concatenate([np.asarray(tr.samples[i]["y"]) for i in range(0,len(tr.samples),37)],0)
gmean=allY.mean(0)
for nm,ds in pools.items():
    sub=Sub(ds,a.per_geom)
    Y=np.concatenate([np.asarray(sub.ds.samples[j]["y"]) for j in sub.keep],0)
    ch=np.sqrt(((Y-gmean)**2).sum(-1)).mean()
    print(f"  chance (predict global-mean source pos) {nm} MAE = {ch:.1f} m",flush=True)
for c in [int(x) for x in a.ctx.split(",")]:
    out=[]
    for nm,ds in pools.items():
        sub=Sub(ds,a.per_geom)
        gm=E.per_geometry_mae(model,conv,sub,dev,c,1,y_mean=ym,y_std=ys,ctx_sample_mode=csm,exclude_ctx_from_target=exc)
        reg={}
        for g,m in gm.items(): reg.setdefault(labels.get(g,{}).get("region","train"),[]).append(m)
        out.append(f"{nm} MAE={float(np.mean(list(gm.values()))):7.2f} ("+" ".join(f"{k}={np.mean(v):.0f}" for k,v in sorted(reg.items()))+")")
    print(f"  ctx={c:3d}: "+"  ".join(out),flush=True)
