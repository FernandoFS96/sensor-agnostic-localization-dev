"""Cierra todos los huecos que quedan para la primera version del paper.

Orden deliberado: lo critico primero, la cola larga al final, para que los
resultados que bloquean el esqueleto esten en unas horas y no detras de 24 h de
entrenamiento del peor baseline.

  1. B2 con el parche de region: pathD etiqueta sus geometrias retenidas
     'deep'/'shallow' (fija el layout y varia el entorno, asi que el eje con
     sentido es la profundidad, no el desplazamiento). ev_region_mae solo emitia
     interp/extrap y tiraba ese desglose.
  2. §4.1 con TODAS las semillas disponibles: la tabla ancha venia de una suite
     vieja con 2 checkpoints por modelo mientras el resto del paper va con 3.
     Esto no entrena nada, solo re-agrega lo que ya existe.
  3. Ablacion del codificador, material de anexo, misma particion y coste bajo.
  4. CL_A_ranp semillas 1 y 2 (12 h cada una, en serie) + su grupo de evaluacion.
     Es el unico modelo de B1 con una sola semilla; su 63,23 +- 0,00 son tres
     semillas de EVALUACION sobre el mismo checkpoint.
"""
import glob, os, re, subprocess, csv, collections, sys
REPO="/home/fer/tesis/sensor-agnostic-underwater-localization"
PYB="/home/fer/miniconda3/envs/torch/bin/python"
OUT="output/eval_paper"
os.chdir(REPO)
SH="data/random_task/hermite_shared/processed/geometry_split"

def ckpts(exp):
    best={}
    for p in glob.glob(f"output/hydra/training/*/{exp}/seed_*/best.pt"):
        s=int(p.split("/seed_")[1].split("/")[0])
        d=re.search(r"/training/([\d-]+)/",p).group(1)
        if s not in best or d>best[s][0]: best[s]=(d,p)
    return [p for _,p in sorted(best.values())]

def suite(group, mods, ddir=None, tiers="1,2,3,4"):
    args=[]
    for lbl,exp in mods.items():
        for p in ckpts(exp): args.append(f"{lbl}={p}")
    if not args:
        print(f"  !! {group}: sin checkpoints",flush=True); return False
    cmd=[PYB,"-u","scripts/eval/eval_suite.py","--models",*args,
         "--tiers",tiers,"--out-dir",f"{OUT}/{group}"]
    if ddir: cmd+=["--data-dir",ddir]
    n=collections.Counter(a.split("=")[0] for a in args)
    print(f"\n--- {group}: {dict(n)}  tiers={tiers}",flush=True)
    with open(f"{OUT}/{group}.log","w") as lf:
        rc=subprocess.call(cmd,stdout=lf,stderr=subprocess.STDOUT)
    print(f"    rc={rc}",flush=True)
    if rc:
        for ln in open(f"{OUT}/{group}.log").read().splitlines()[-10:]:
            print(f"      {ln}",flush=True)
    return rc==0

def train(exp, seed, overrides):
    lg=f"scratch/ab_traj/{exp}_s{seed}.log"
    print(f"\n--- entrenando {exp} semilla {seed}  "
          f"{__import__('time').strftime('%F %T')}",flush=True)
    with open(lg,"w") as f:
        rc=subprocess.call([PYB,"scripts/train/train_np_geometry.py",*overrides,
                            f"seed={seed}",f"exp_name={exp}"],
                           stdout=f,stderr=subprocess.STDOUT,
                           env={**os.environ,"TQDM_DISABLE":"1"})
    print(f"    rc={rc}",flush=True)
    for ln in open(lg).read().splitlines():
        if "early stop" in ln or "done. best val mae" in ln: print(f"      {ln}",flush=True)
    return rc==0

# ---- 1. B2 con desglose deep/shallow ------------------------------------------
suite("b2_prefix", {"cnp":"PD_A_cnp","cnp_ffn":"PD_A_cnpF","pfn":"PD_A_pfn","rcnp":"PD_A_rcnp"})
suite("b2_calib",  {"cnp":"PD_B_cnp","cnp_ffn":"PD_B_cnpF","pfn":"PD_B_pfn"})

# ---- 2-3. §4.1 y ablacion, todas las semillas ---------------------------------
suite("s4_family", {"flat_cnp":"paper_flat_cnp","flat_anp":"paper_flat_anp",
                    "spatial_cnp":"paper_spatial_cnp_full","spatial_anp":"paper_spatial_anp",
                    "rcnp":"paper_rcnp","ranp":"paper_ranp",
                    "spatial_ranp":"paper_spatial_ranp"}, SH)
suite("s4_ablation",{"full":"paper_spatial_cnp_full","no_tokenize":"paper_abl_notokenize",
                     "no_position":"paper_abl_nopos","no_attention":"paper_abl_noattn",
                     "mean_pool":"paper_abl_meanpool"}, SH)

# ---- 4. RANP de B1: dos semillas mas, en serie --------------------------------
RANP_OV=["data.data_dir=data/pfn10kd/hermite_distinct/processed/geometry_split",
         "data.max_train_geometries=null","data.max_train_trajectories=null",
         "data.max_val_trajectories=null","wandb.enabled=false",
         "model=spatial_ranp","data.context.per_sample=false"]
for s in (1,2):
    train("CL_A_ranp", s, RANP_OV)
suite("b1_ranp", {"ranp":"CL_A_ranp"})

# ---- resumen ------------------------------------------------------------------
print("\n\n================ RESUMEN FINAL ================",flush=True)
for g in ("b2_prefix","b2_calib","s4_family","s4_ablation","b1_ranp"):
    f=f"{OUT}/{g}/results.csv"
    if not os.path.exists(f): continue
    r=list(csv.DictReader(open(f)))
    regs=[m for m in sorted({x["metric"] for x in r if x["test"]=="region_mae"})
          if m not in ("overall","train")]
    print(f"\n  === {g}",flush=True)
    hdr=f"  {'modelo':<16}{'MAE':>15}" + "".join(f"{x:>10}" for x in regs) + f"{'noise':>8}{'shuffle':>9}{'n':>4}"
    print(hdr,flush=True)
    for m in sorted({x["model"] for x in r}):
        p=lambda te,me: next((x for x in r if x["model"]==m and x["test"]==te and x["metric"]==me),None)
        o=p("region_mae","overall")
        fm=lambda z: f"{float(z['mean']):.2f}" if z else "-"
        mae=f"{float(o['mean']):.2f} ± {float(o['std']):.2f}" if o else "-"
        line=f"  {m:<16}{mae:>15}" + "".join(f"{fm(p('region_mae',x)):>10}" for x in regs)
        print(line+f"{fm(p('acoustic_reliance','noise_ratio')):>8}"
                   f"{fm(p('temporal_shuffle','shuffle_ratio')):>9}{(o['n'] if o else '-'):>4}",flush=True)
print("\n  TODO EN output/eval_paper/<grupo>/results.csv",flush=True)
