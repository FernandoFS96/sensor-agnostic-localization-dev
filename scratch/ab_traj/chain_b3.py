"""Reintento de B3 con niveles 1-3.

El nivel 4 (permutation + sensor_dropout) manipula sensores de uno en uno y exige
`sensor_mask`, que solo soporta el SpatialEncoder tokenizado. TODOS los modelos de
B3 se entrenaron con tokenize=false (es el control plano del eje sin posiciones),
asi que el nivel 4 no es que falle: no es aplicable. La invarianza a permutacion
es una propiedad del frontal tokenizado y no significa nada aqui.
"""
import glob, os, re, subprocess, csv, collections
REPO="/home/fer/tesis/sensor-agnostic-underwater-localization"
PY_BIN="/home/fer/miniconda3/envs/torch/bin/python"
OUT="output/eval_paper"
os.chdir(REPO)

def ckpts(exp):
    best={}
    for p in glob.glob(f"output/hydra/training/*/{exp}/seed_*/best.pt"):
        s=int(p.split("/seed_")[1].split("/")[0])
        d=re.search(r"/training/([\d-]+)/",p).group(1)
        if s not in best or d>best[s][0]: best[s]=(d,p)
    return [p for _,p in sorted(best.values())]

CELLS=["g100_t10","g400_t10","g1600_t10","g4000_t10","g8000_t10","g1600_t50","g8000_t50"]
done,failed=[],[]
for i,cell in enumerate(CELLS,1):
    args=[]
    for pref,lbl in (("PC","sin_posiciones"),("PA","con_posiciones"),("PAC","con_pos_notoken")):
        for p in ckpts(f"{pref}_{cell}"): args.append(f"{lbl}={p}")
    if not args: print(f"  !! b3_{cell}: sin checkpoints",flush=True); continue
    g=f"b3_{cell}"
    cmd=[PY_BIN,"-u","scripts/eval/eval_suite.py","--models",*args,
         "--tiers","1,2,3","--out-dir",f"{OUT}/{g}",
         "--data-dir",f"data/calib_10k/cells/{cell}"]
    n=collections.Counter(a.split("=")[0] for a in args)
    print(f"\n--- [{i}/{len(CELLS)}] {g}: {dict(n)}",flush=True)
    with open(f"{OUT}/{g}.log","w") as lf:
        rc=subprocess.call(cmd,stdout=lf,stderr=subprocess.STDOUT)
    print(f"    rc={rc}",flush=True)
    (done if rc==0 else failed).append(g)
    if rc: 
        for ln in open(f"{OUT}/{g}.log").read().splitlines()[-8:]: print(f"      {ln}",flush=True)

print("\n\n============ B3 — CURVA DE ESCALA ============",flush=True)
print(f"\n  {'celda':<12}{'brazo':<18}{'MAE':>15}{'noise':>8}{'shuffle':>9}{'n':>4}",flush=True)
for cell in CELLS:
    f=f"{OUT}/b3_{cell}/results.csv"
    if not os.path.exists(f): continue
    r=list(csv.DictReader(open(f)))
    for m in sorted({x["model"] for x in r}):
        p=lambda te,me: next((x for x in r if x["model"]==m and x["test"]==te and x["metric"]==me),None)
        o=p("region_mae","overall")
        fm=lambda z: f"{float(z['mean']):.2f}" if z else "-"
        mae=f"{float(o['mean']):.2f} ± {float(o['std']):.2f}" if o else "-"
        print(f"  {cell:<12}{m:<18}{mae:>15}{fm(p('acoustic_reliance','noise_ratio')):>8}"
              f"{fm(p('temporal_shuffle','shuffle_ratio')):>9}{(o['n'] if o else '-'):>4}",flush=True)
if failed: print(f"\n  !! fallos: {failed}",flush=True)
