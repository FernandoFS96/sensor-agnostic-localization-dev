#!/usr/bin/env bash
# Reparacion de la semilla 0, que el OOM killer mato el 1 de octubre a las 15:45.
#
# ARRANQUE ESCALONADO, y no por cautela generica: la traza del vigilante muestra
# que la oleada 1 cargo los dos pickles de 48 GB A LA VEZ y quedo clavada en
# 124 G usados / 0 G disponibles / swap 7-7, congelada media hora antes de morir.
# Las oleadas 2 y 3 arrancaron desfasadas por como las encadena `wait`, y con una
# run ya en regimen mientras la otra cargaba el sistema se quedo en 98 G con 26 G
# libres y swap a 0; ambas sobrevivieron 6-8 h. El pico de la carga simultanea es
# lo que envenena, no el regimen estacionario: basta no solaparlos.
#
# El segundo arranca cuando el primero imprime `device=cuda`, que es la primera
# linea posterior a la carga del pickle. Un `sleep` fijo seria una apuesta sobre
# la velocidad del disco; esto es la senal real.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/random_task/hermite_distinct/processed_common/geometry_split
LOG=scratch/ab_traj

BASE="data=layout_ood dataset=random_hermite_distinct training.num_workers=4
      wandb.offline=true wandb.enabled=false
      model.num_hidden=128 data.context.per_sample=false data.data_dir=$DD"

launch () {   # $1=exp $2=model $3=aug
  echo "  -> $1 s0  $(date '+%T')"
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 data.trans_aug_m=$3 seed=0 exp_name=$1 > $LOG/$1_s0.log 2>&1 &
}

echo "=== RERUN SEMILLA 0 — INICIO $(date '+%F %T') ==="
: > $LOG/AB_dist_spatial_cnp_s0.log
: > $LOG/AB_dist_rcnp_s0.log
launch AB_dist_spatial_cnp spatial_cnp 150
# esperar a que el primero pase de la carga (tope 40 min por si algo falla)
for i in $(seq 1 160); do
  grep -q "device=cuda" $LOG/AB_dist_spatial_cnp_s0.log 2>/dev/null && break
  sleep 15
done
echo "  spatial_cnp ya en GPU a las $(date '+%T'); $(free -g | awk 'NR==2{print "disp="$7"G"}')"
launch AB_dist_rcnp rcnp 0
wait
echo "=== RERUN COMPLETO $(date '+%F %T') ==="
for t in AB_dist_spatial_cnp AB_dist_rcnp; do
  grep -E "early stop|done\. best val mae" $LOG/${t}_s0.log | sed "s/^/  /"
done
