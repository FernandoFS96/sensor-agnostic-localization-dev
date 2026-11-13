"""Q3 part 1: inference-time position ablation (fast: val+test, subsampled)."""
import os,sys,json,argparse,numpy as np, torch as t
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec=importlib.util.spec_from_file_location("evalmod","scripts/eval/eval_np_geometry.py")
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--eval_ctx",type=int,default=10)
ap.add_argument("--draws",type=int,default=1); ap.add_argument("--per_geom",type=int,default=40)
ap.add_argument("--pools",default="val,test"); a=ap.parse_args()
dev=t.device("cuda" if t.cuda.is_available() else "cpu")
ck=t.load(a.ckpt,map_location=dev); cfg=ck.get("config",{}); name=ck.get("model_name","cnp"); conv=ck.get("convention","split")
data_dir=cfg.get("data_dir")
if not os.path.isabs(data_dir): data_dir=os.path.normpath(os.path.join(".",data_dir))
sc=cfg.get("spatial",None)
model,conv2=E.build_model(name,cfg.get("num_hidden",128),ck.get("feat_dim"),ck.get("out_dim",3),dropout=cfg.get("dropout",0.1),spatial_cfg=sc)
conv=conv or conv2; model.to(dev); model.load_state_dict(ck["model"]); model.eval()
ym=ck.get("y_mean"); ys=ck.get("y_std")
csm=cfg.get("ctx_sample_mode","first"); exc=bool(cfg.get("exclude_ctx_from_target",True))
labels={}; sp_path=os.path.join(data_dir,"splits.json")
if os.path.exists(sp_path): labels={int(k):v for k,v in json.load(open(sp_path)).get("labels",{}).items()}
rng=np.random.default_rng(0)
class Sub:
    def __init__(s,ds,per_geom,zero=False):
        s.zero=zero
        by={}
        for i,smp in enumerate(ds.samples): by.setdefault(int(smp.get("geometry_id",-1)),[]).append(i)
        keep=[]
        for g,idxs in by.items():
            idxs=np.array(idxs); sel=idxs if len(idxs)<=per_geom else rng.choice(idxs,per_geom,replace=False)
            keep+=list(sel)
        s.ds=ds; s.keep=keep
    def __len__(s): return len(s.keep)
    def __getitem__(s,i):
        X,y,g,th,spv=s.ds[s.keep[i]]
        if s.zero and spv.numel()>0: spv=t.zeros_like(spv)
        return X,y,g,th,spv
pool_names=a.pools.split(",")
pools={nm:E.TrajectoryDataset(os.path.join(data_dir,f"{nm}_data.pkl")) for nm in pool_names if os.path.exists(os.path.join(data_dir,f"{nm}_data.pkl"))}
print(f"ckpt={a.ckpt}",flush=True)
print(f"  epoch={ck.get('epoch')} use_position={sc.get('use_position') if sc else None} aug={cfg.get('trans_aug_m','?')} data={data_dir}",flush=True)
print(f"  eval_ctx={a.eval_ctx} draws={a.draws} per_geom={a.per_geom} csm={csm} exclude={exc}",flush=True)
def run(zero):
    res={}
    for nm,ds in pools.items():
        sub=Sub(ds,a.per_geom,zero=zero)
        gm=E.per_geometry_mae(model,conv,sub,dev,a.eval_ctx,a.draws,y_mean=ym,y_std=ys,ctx_sample_mode=csm,exclude_ctx_from_target=exc)
        res[nm]=float(np.mean(list(gm.values())))
        reg={}
        for g,m in gm.items(): reg.setdefault(labels.get(g,{}).get("region","train"),[]).append(m)
        print(f"  [{'zero' if zero else 'pos '}] {nm:5s} MAE={res[nm]:7.2f}   "+"  ".join(f"{k}={np.mean(v):.1f}(n{len(v)})" for k,v in reg.items()),flush=True)
    return res
print("\n-- TRUE sensor_pos (ORACLE) --",flush=True); true=run(False)
print("\n-- ZEROED sensor_pos (position removed at inference) --",flush=True); zero=run(True)
print("\n== SUMMARY ==",flush=True)
for nm in pools: print(f"  {nm:5s}: oracle={true[nm]:.2f}  no-pos={zero[nm]:.2f}  delta=+{zero[nm]-true[nm]:.2f} ({100*(zero[nm]-true[nm])/true[nm]:+.0f}%)",flush=True)
