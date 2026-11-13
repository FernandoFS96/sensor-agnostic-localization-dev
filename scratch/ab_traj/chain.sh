#!/usr/bin/env bash
# Espera al rerun de la semilla 0 y encadena detras la suite de evaluacion.
# Un solo proceso desatendido: cuando termine, las cinco tablas del paper estan
# en la MISMA metrica (region MAE del pool retenido) y el esqueleto se rellena
# de una pasada.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
RERUN_PID=${1:-}

if [ -n "$RERUN_PID" ]; then
  echo "=== esperando al rerun (pid $RERUN_PID) — $(date '+%F %T') ==="
  while kill -0 "$RERUN_PID" 2>/dev/null; do sleep 60; done
  echo "=== rerun terminado $(date '+%F %T') ==="
  cat scratch/ab_traj/rerun_s0_out.log | sed 's/^/  /'
fi

# La semilla 0 solo entra en el A/B si converge. Si volviese a morir, el grupo
# ab_distinct se evalua con las semillas 1 y 2 y la cadena sigue: perder una
# semilla no justifica perder las otras doce tablas.
echo ""
echo "=== estado de la semilla 0 ==="
for t in AB_dist_spatial_cnp AB_dist_rcnp; do
  if grep -q "done. best val mae" scratch/ab_traj/${t}_s0.log 2>/dev/null; then
    echo "  [CONVERGIDA] $(grep -h 'done. best val mae' scratch/ab_traj/${t}_s0.log | sed 's/.*best val mae=/'"$t"' best=/;s/ ->.*//')"
  else
    echo "  [NO CONVERGIDA] $t s0 — queda fuera del A/B"
  fi
done

echo ""
$PY -u scratch/ab_traj/chain_eval.py
echo ""
echo "=== CADENA COMPLETA $(date '+%F %T') ==="
