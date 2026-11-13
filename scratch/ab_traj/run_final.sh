#!/usr/bin/env bash
# Espera a que B3 termine y encadena detras todo lo que falta.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
B3_PID=${1:-}
if [ -n "$B3_PID" ]; then
  echo "=== esperando a B3 (pid $B3_PID) $(date '+%F %T') ==="
  while kill -0 "$B3_PID" 2>/dev/null; do sleep 60; done
  echo "=== B3 terminado $(date '+%F %T') ==="
  tail -25 scratch/ab_traj/b3_out.log | sed 's/^/  /'
fi
echo ""
/home/fer/miniconda3/envs/torch/bin/python -u scratch/ab_traj/chain_final.py
echo ""
echo "=== TODO COMPLETO $(date '+%F %T') ==="
