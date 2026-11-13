#!/usr/bin/env bash
# A/B: ¿cuánto afecta que cada despliegue tenga SUS PROPIAS trayectorias?
#
# Único factor que cambia: el modo de trayectoria.
#   brazo SHARED   -> data/random_task/hermite_shared/processed            (ya entrenado)
#                     paper_spatial_cnp_full 24,10 ± 0,41  ·  paper_rcnp 27,22 ± 1,21
#   brazo DISTINCT -> data/random_task/hermite_distinct/processed_common   (este script)
#
# `processed_common` se reprocesó con el splits.json de `shared`, porque las
# particiones originales diferían (160/20/20 frente a 140/30/20) y eso habría
# mezclado dos factores. Ahora la partición de geometrías es idéntica.
#
# Cada modelo conserva SU configuración original, copiada de los overrides de
# referencia. En particular rcnp lleva trans_aug_m=0 y spatial_cnp lleva 150: no es
# un descuido, el aumento por traslación exige un modelo espacial y rcnp no lo es.
# La pregunta no es qué modelo gana, sino cuánto se mueve cada uno al cambiar el eje.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/random_task/hermite_distinct/processed_common/geometry_split
LOG=scratch/ab_traj; mkdir -p $LOG
MAXJOBS=2        # ~48 GB residentes por run, de 125 GB

# `dataset=random_hermite_distinct` queda solo como identidad en el checkpoint:
# data.data_dir la sobrescribe, porque la ruta interpolada apuntaría a `processed`
# (partición 140/30/20) en lugar de a `processed_common`.
BASE="data=layout_ood dataset=random_hermite_distinct training.num_workers=4
      wandb.offline=true wandb.enabled=false
      model.num_hidden=128 data.context.per_sample=false data.data_dir=$DD"

run () {   # $1=exp $2=model $3=aug $4=seed
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
  echo "  -> $1 s$4  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 data.trans_aug_m=$3 seed=$4 exp_name=$1 > $LOG/$1_s$4.log 2>&1 &
}

echo "=== A/B TRAYECTORIAS — INICIO $(date '+%F %T') ==="
for S in 0 1 2; do
  run AB_dist_spatial_cnp spatial_cnp 150 $S
  run AB_dist_rcnp        rcnp           0 $S
done
wait
echo "=== COMPLETO $(date '+%F %T') ==="

$PY - <<'EOF'
import csv, glob, os, numpy as np
def best(tag):
    """Solo runs CONVERGIDAS. El umbral anterior (>=200 epocas) dejaba entrar runs
    que el OOM killer mato a mitad: con el coseno de lr sin decaer su mejor val_mae
    esta muy por encima del real, y mezclaba duracion de entrenamiento con el factor
    que queremos medir. La linea `done. best val mae` solo se imprime al terminar
    (por early stopping o por agotar epocas), asi que es la prueba que vale."""
    v=[]
    for d in glob.glob(f"output/hydra/training/*/{tag}/seed_*"):
        seed=d.split("/seed_")[1]
        lg=f"scratch/ab_traj/{tag}_s{seed}.log"
        if not (os.path.exists(lg) and "done. best val mae" in open(lg).read()):
            continue
        try:
            r=[x for x in csv.DictReader(open(f"{d}/train_log.csv")) if x.get("val_mae")]
            if r: v.append(min(float(x["val_mae"]) for x in r))
        except Exception: pass
    return v
print("\n  EFECTO DEL MODO DE TRAYECTORIA — MAE de validación, misma partición\n")
print(f"  {'modelo':<16}{'compartidas':>20}{'propias':>20}{'Δ':>10}")
for lbl,sh,di in (("Spatial CNP","paper_spatial_cnp_full","AB_dist_spatial_cnp"),
                  ("RCNP","paper_rcnp","AB_dist_rcnp")):
    a,b=best(sh),best(di)
    fa=f"{np.mean(a):.2f} ± {np.std(a):.2f}" if a else "-"
    fb=f"{np.mean(b):.2f} ± {np.std(b):.2f}" if b else "-"
    d=f"{np.mean(b)-np.mean(a):+.2f}" if (a and b) else "-"
    print(f"  {lbl:<16}{fa:>20}{fb:>20}{d:>10}")
print("\n  Δ positivo = empeora al dar trayectorias propias a cada despliegue.")
EOF
