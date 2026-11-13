#!/usr/bin/env bash
# Smoke test del A/B: 2 épocas por brazo, SECUENCIAL.
# Secuencial a propósito: cada run mantiene residente el pickle de 48 GB y dos a
# la vez durante la carga rozan los 125 GB. El run completo sí va con MAXJOBS=2,
# pero ahí el objetivo es rendimiento, no descartar un fallo de arranque.
# Qué tiene que demostrar: que el modelo construye, que lee processed_common, que
# una época entera de train+val termina y que val_mae aterriza en train_log.csv.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
DD=data/random_task/hermite_distinct/processed_common/geometry_split
LOG=scratch/ab_traj

BASE="data=layout_ood dataset=random_hermite_distinct training.num_workers=4
      wandb.offline=true wandb.enabled=false
      model.num_hidden=128 data.context.per_sample=false data.data_dir=$DD"

rc_all=0
for spec in "SMOKE_spatial_cnp spatial_cnp 150" "SMOKE_rcnp rcnp 0"; do
  set -- $spec
  echo "=== $1  $(date '+%T') ==="
  TQDM_DISABLE=1 $PY scripts/train/train_np_geometry.py $BASE \
      model=$2 data.trans_aug_m=$3 seed=0 exp_name=$1 \
      training.epochs=2 > $LOG/$1.log 2>&1
  rc=$?; echo "  rc=$rc"
  [ $rc -ne 0 ] && { rc_all=1; tail -25 $LOG/$1.log | sed 's/^/    /'; continue; }
  d=$(ls -d output/hydra/training/*/$1/seed_0 2>/dev/null | tail -1)
  echo "  out: $d"
  [ -f "$d/train_log.csv" ] && cat "$d/train_log.csv" | cut -d, -f1,5,9,11 | sed 's/^/    /'
  [ -f "$d/best.pt" ] && echo "    best.pt $(du -h $d/best.pt | cut -f1)" || { echo "    !! sin best.pt"; rc_all=1; }
done
echo "=== SMOKE rc_all=$rc_all  $(date '+%T') ==="
exit $rc_all
