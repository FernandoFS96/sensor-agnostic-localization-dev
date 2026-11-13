"""CALIBRATION regime (spec 3.3): context points come from a DIFFERENT trajectory
than the targets, same geometry. Trajectory memorization cannot help, so any
performance must come from acoustics + in-context geometry inference.

Reports, per checkpoint:  RANDOM/PREFIX (same-traj, current protocol) vs
CALIBRATION (cross-traj), each with real vs scrambled acoustics."""
import os,sys,json,argparse,numpy as np, torch as t
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec=importlib.util.spec_from_file_location("evalmod","scripts/eval/eval_np_geometry.py")
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--ctx",type=int,default=40); ap.add_argument("--pairs",type=int,default=20)
ap.add_argument("--tag",default=""); a=ap.parse_args()
dev=t.device("cuda" if t.cuda.is_available() else "cpu")
ck=t.load(a.ckpt,map_location=dev); cfg=ck.get("config",{})
name=ck.get("model_name","cnp"); conv=ck.get("convention","split")
dd=cfg.get("data_dir")
if not os.path.isabs(dd): dd=os.path.normpath(os.path.join(".",dd))
sc=cfg.get("spatial",None)
model,_=E.build_model(name,cfg.get("num_hidden",128),ck.get("feat_dim"),ck.get("out_dim",3),dropout=cfg.get("dropout",0.1),spatial_cfg=sc)
model.to(dev); model.load_state_dict(ck["model"]); model.eval()
ym=ck.get("y_mean"); ys=ck.get("y_std")
ymd=ym.to(dev) if ym is not None else None; ysd=ys.to(dev) if ys is not None else None
ds=E.TrajectoryDataset(os.path.join(dd,"val_data.pkl"))
labels={}; spj=os.path.join(dd,"splits.json")
if os.path.exists(spj): labels={int(k):v for k,v in json.load(open(spj)).get("labels",{}).items()}
by={}
for i,s in enumerate(ds.samples): by.setdefault(int(s.get("geometry_id",-1)),[]).append(i)
rng=np.random.default_rng(0)

@t.no_grad()
def run(cross, scramble):
    errs={}
    for g,idxs in by.items():
        idxs=np.array(idxs); es=[]
        for _ in range(a.pairs):
            i=int(rng.choice(idxs))
            j=int(rng.choice(idxs[idxs!=i])) if cross else i
            Xc,yc,_,_,spv=ds[i]      # context donor trajectory
            Xt,yt,_,_,_   =ds[j]     # target trajectory
            if scramble:             # acoustics from an unrelated sample
                k=int(rng.integers(len(ds.samples)))
                Xd,_,_,_,_=ds[k]; Xc=Xd
                k2=int(rng.integers(len(ds.samples)))
                Xd2,_,_,_,_=ds[k2]; Xt=Xd2
            n=min(a.ctx,Xc.size(0))
            cx=Xc[:n].unsqueeze(0).to(dev); cy=yc[:n].unsqueeze(0).to(dev)
            if cross: tsel=t.arange(Xt.size(0))            # score ALL target-traj points
            else:     tsel=t.arange(n,Xt.size(0))          # complement (current protocol)
            if tsel.numel()==0: continue
            tx=Xt[tsel].unsqueeze(0).to(dev); ty=yt[tsel].unsqueeze(0).to(dev)
            spt=spv.unsqueeze(0).to(dev) if spv.numel()>0 else None
            cyn=(cy-ymd)/ysd if ymd is not None else cy
            mean,var,*_=model(cx,cyn,tx,None,sensor_pos=spt)
            if ymd is not None: mean=mean*ysd+ymd
            es.append(float(t.sqrt(((mean-ty)**2).sum(-1)+1e-12).mean()))
        if es: errs[g]=float(np.mean(es))
    reg={}
    for g,m in errs.items(): reg.setdefault(labels.get(g,{}).get("region","train"),[]).append(m)
    return float(np.mean(list(errs.values()))), {k:float(np.mean(v)) for k,v in reg.items()}

print(f"[{a.tag}] use_position={sc.get('use_position') if sc else None} ctx={a.ctx} pairs/geom={a.pairs}",flush=True)
for cross,lab in [(False,"SAME-TRAJ (current protocol)"),(True,"CALIBRATION (cross-traj)")]:
    for scr,slab in [(False,"real acoustics"),(True,"SCRAMBLED acoustics")]:
        m,reg=run(cross,scr)
        print(f"  {lab:30s} {slab:20s}: MAE={m:7.2f}  "+" ".join(f"{k}={v:.0f}" for k,v in sorted(reg.items())),flush=True)
