"""Pasa la suite de evaluacion por todo lo que necesita el paper, en serie.

Por que en serie y no en paralelo: cada invocacion carga su propio pool de
geometrias retenidas (pfn10kd son 2000) y dos a la vez repetirian el error que
mato la semilla 0 del A/B el 1 de octubre.

Seleccion de checkpoints: para cada experimento, TODAS las semillas, y para cada
semilla el directorio de fecha MAS RECIENTE. Esto importa porque los seed_0 del
A/B que el OOM killer mato siguen en disco con su best.pt de la epoca 279, bajo
2026-10-01, junto al rerun bueno bajo 2026-10-03; coger el mas reciente descarta
el muerto. Para los grupos marcados `require_done` se exige ademas que el log
diga `done. best val mae`, que solo se imprime al terminar de verdad.
"""
import glob, os, re, subprocess, sys, json, csv
import collections

REPO = "/home/fer/tesis/sensor-agnostic-underwater-localization"
PY_BIN = "/home/fer/miniconda3/envs/torch/bin/python"
OUT = "output/eval_paper"
os.chdir(REPO)

def ckpts(exp, require_done=False):
    """{semilla: ruta} con el directorio de fecha mas reciente por semilla."""
    best = {}
    for p in glob.glob(f"output/hydra/training/*/{exp}/seed_*/best.pt"):
        m = re.search(r"/training/([\d-]+)/", p)
        seed = int(p.split("/seed_")[1].split("/")[0])
        date = m.group(1) if m else ""
        if seed not in best or date > best[seed][0]:
            best[seed] = (date, p)
    if require_done:
        keep = {}
        for s, (d, p) in best.items():
            lg = f"scratch/ab_traj/{exp}_s{s}.log"
            if os.path.exists(lg) and "done. best val mae" in open(lg).read():
                keep[s] = (d, p)
            else:
                print(f"    [descartada] {exp} s{s}: sin 'done.' en {lg}", flush=True)
        best = keep
    return {s: p for s, (d, p) in sorted(best.items())}

# grupo -> (etiqueta_modelo -> experimento, data_dir o None, require_done)
GROUPS = [
 ("ab_shared",  {"spatial_cnp":"paper_spatial_cnp_full", "rcnp":"paper_rcnp"},
                "data/random_task/hermite_shared/processed/geometry_split", False),
 ("ab_distinct",{"spatial_cnp":"AB_dist_spatial_cnp", "rcnp":"AB_dist_rcnp"},
                "data/random_task/hermite_distinct/processed_common/geometry_split", True),
 ("b1_prefix",  {"cnp":"CL_A_cnp","cnp_ffn":"CL_A_cnpF","pfn":"CL_A_pfn",
                 "rcnp":"CL_A_rcnp","ranp":"CL_A_ranp"}, None, False),
 ("b1_calib",   {"cnp":"CL_B_cnp","cnp_ffn":"CL_B_cnpF","pfn":"CL_B_pfn"}, None, False),
 ("b2_prefix",  {"cnp":"PD_A_cnp","cnp_ffn":"PD_A_cnpF","pfn":"PD_A_pfn",
                 "rcnp":"PD_A_rcnp"}, None, False),
 ("b2_calib",   {"cnp":"PD_B_cnp","cnp_ffn":"PD_B_cnpF","pfn":"PD_B_pfn"}, None, False),
]
# B3: una invocacion por celda, porque cada celda es su propio data_dir. El pool
# retenido (val_data.pkl) esta enlazado a _shared, asi que la curva de escala se
# mide sobre el MISMO conjunto en todas las celdas. No hay splits.json en las
# celdas, de modo que region_mae dara `overall` pero no interp/extrap.
for cell in ["g100_t10","g400_t10","g1600_t10","g4000_t10","g8000_t10",
             "g1600_t50","g8000_t50"]:
    mods = {}
    for pref, lbl in (("PC","sin_posiciones"), ("PA","con_posiciones"), ("PAC","con_pos_notoken")):
        e = f"{pref}_{cell}"
        if glob.glob(f"output/hydra/training/*/{e}/seed_*/best.pt"):
            mods[lbl] = e
    if mods:
        GROUPS.append((f"b3_{cell}", mods, f"data/calib_10k/cells/{cell}", False))

os.makedirs(OUT, exist_ok=True)
print(f"=== CADENA DE EVALUACION — {len(GROUPS)} grupos ===", flush=True)
done_groups, failed = [], []
for gi, (g, mods, ddir, req) in enumerate(GROUPS, 1):
    args = []
    for lbl, exp in mods.items():
        cs = ckpts(exp, req)
        if not cs:
            print(f"  !! {g}: {exp} sin checkpoints usables", flush=True); continue
        args += [f"{lbl}={p}" for p in cs.values()]
    if not args:
        print(f"  !! {g}: nada que evaluar, salto", flush=True); failed.append(g); continue
    od = f"{OUT}/{g}"
    cmd = [PY_BIN, "-u", "scripts/eval/eval_suite.py", "--models", *args,
           "--tiers", "1,2,3,4", "--out-dir", od]
    if ddir: cmd += ["--data-dir", ddir]
    nseeds = collections.Counter(a.split("=")[0] for a in args)
    print(f"\n--- [{gi}/{len(GROUPS)}] {g}: {dict(nseeds)}", flush=True)
    with open(f"{OUT}/{g}.log", "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    print(f"    rc={rc}", flush=True)
    if rc == 0: done_groups.append(g)
    else:
        failed.append(g)
        print("    ultimas lineas:", flush=True)
        for ln in open(f"{OUT}/{g}.log").read().splitlines()[-12:]:
            print(f"      {ln}", flush=True)

# ---- tabla consolidada --------------------------------------------------------
print("\n\n==================== RESUMEN ====================", flush=True)
for g in done_groups:
    f = f"{OUT}/{g}/results.csv"
    if not os.path.exists(f): continue
    r = list(csv.DictReader(open(f)))
    pick = lambda m, k: next((x for x in r if x["model"]==m and x["test"]==k[0]
                              and x["metric"]==k[1]), None)
    models = sorted({x["model"] for x in r})
    print(f"\n  === {g}")
    print(f"  {'modelo':<18}{'MAE':>14}{'interp':>9}{'extrap':>9}"
          f"{'noise':>8}{'shuffle':>9}{'n':>4}")
    for m in models:
        o  = pick(m, ("region_mae","overall"))
        it = pick(m, ("region_mae","interp")); ex = pick(m, ("region_mae","extrap"))
        nz = pick(m, ("acoustic_reliance","noise_ratio"))
        sh = pick(m, ("temporal_shuffle","shuffle_ratio"))
        fm = lambda z, d=2: f"{float(z['mean']):.{d}f}" if z else "-"
        mae = f"{float(o['mean']):.2f} ± {float(o['std']):.2f}" if o else "-"
        print(f"  {m:<18}{mae:>14}{fm(it):>9}{fm(ex):>9}{fm(nz):>8}{fm(sh):>9}"
              f"{(o['n'] if o else '-'):>4}")
if failed:
    print(f"\n  !! grupos con fallo: {failed}", flush=True)
print(f"\n  CSVs en {OUT}/<grupo>/results.csv (+ results_per_seed.csv)", flush=True)
