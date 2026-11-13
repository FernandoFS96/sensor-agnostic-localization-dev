#!/usr/bin/env bash
# Cierra la semilla 0 de forma ESTRICTAMENTE SECUENCIAL y encadena la evaluacion.
#
# Por que secuencial y no escalonado: el escalonado solo rebaja el pico de carga,
# no el regimen. Con dos runs residentes el sistema se queda en ~123 G usados y
# 1-2 G disponibles, y el consumo sigue creciendo durante horas porque los 4
# workers del dataloader tocan las cabeceras de los objetos del pickle al iterar
# y van copiando paginas por copy-on-write. Eso es lo que mato la semilla 0 el 1
# de octubre a las 3,5 h, y el intento escalonado de hoy iba por el mismo camino
# (swap 0 -> 2 G en 20 min). Con una sola run: 62 G usados, 62 G libres.
#
# No se toca ninguna config (num_workers incluido): bajarlo rebajaria la memoria,
# pero dejaria la semilla 0 entrenada distinto que las semillas 1 y 2 y la media
# del brazo mezclaria dos cosas. Se paga en tiempo, no en comparabilidad.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/random_task/hermite_distinct/processed_common/geometry_split
LOG=scratch/ab_traj
SPATIAL_PID=${1:?falta el pid del spatial_cnp en marcha}

BASE="data=layout_ood dataset=random_hermite_distinct training.num_workers=4
      wandb.offline=true wandb.enabled=false
      model.num_hidden=128 data.context.per_sample=false data.data_dir=$DD"

echo "=== SECUENCIAL — INICIO $(date '+%F %T') ==="
echo "  1/3 esperando a AB_dist_spatial_cnp s0 (pid $SPATIAL_PID, ya en marcha)"
while kill -0 "$SPATIAL_PID" 2>/dev/null; do sleep 60; done
echo "  spatial_cnp s0 terminado $(date '+%T')"
grep -E "early stop|done\. best val mae" $LOG/AB_dist_spatial_cnp_s0.log | sed 's/^/      /'

echo ""
echo "  2/3 AB_dist_rcnp s0  $(date '+%T')   $(free -g | awk 'NR==2{print "disp="$7"G"}')"
: > $LOG/AB_dist_rcnp_s0.log
TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
    model=rcnp data.trans_aug_m=0 seed=0 exp_name=AB_dist_rcnp \
    > $LOG/AB_dist_rcnp_s0.log 2>&1
echo "  rcnp s0 terminado $(date '+%T') rc=$?"
grep -E "early stop|done\. best val mae" $LOG/AB_dist_rcnp_s0.log | sed 's/^/      /'

echo ""
echo "  3/3 cadena de evaluacion  $(date '+%F %T')"
$PY -u scratch/ab_traj/chain_eval.py
echo ""
echo "=== TODO COMPLETO $(date '+%F %T') ==="
