#!/usr/bin/env bash
# Vigilante pasivo del A/B. No toca nada: solo deja rastro.
# Existe porque el swap quedo a 0 de margen durante la carga simultanea de los dos
# pickles de 48 GB. No hubo thrashing (si/so ~ 0), pero sin margen un pico de
# asignacion lo resuelve el OOM killer, y 35 h sin nadie delante es mucho tiempo
# para enterarse despues sin pruebas.
set -u
cd /home/fer/tesis/sensor-agnostic-underwater-localization
L=scratch/ab_traj/watchdog.log
: > $L
while kill -0 "$1" 2>/dev/null; do
  {
    printf "%s  " "$(date '+%F %T')"
    free -m | awk 'NR==2{printf "mem_usada=%dG disp=%dG  ",$3/1024,$7/1024} NR==3{printf "swap=%dG/%dG  ",$3/1024,$2/1024}'
    # solo los procesos padre: pgrep contaria tambien los workers forkeados del
    # dataloader (4 por run) y diria "runs=20" con dos entrenamientos en marcha.
    printf "runs=%s  " "$(pgrep -P "$1" | wc -l)"
    for f in output/hydra/training/*/AB_dist_*/seed_*/train_log.csv; do
      [ -f "$f" ] || continue
      printf "%s=%s " "$(echo "$f" | sed 's|.*/AB_dist_||;s|/train_log.csv||;s|/seed_|.s|')" "$(( $(wc -l < "$f") - 1 ))"
    done
    echo
  } >> $L
  # cualquier OOM kill sobre un proceso nuestro es la prueba que buscamos
  dmesg -T 2>/dev/null | grep -i "killed process" | tail -3 | sed 's/^/    OOM: /' >> $L
  sleep 300
done
echo "$(date '+%F %T')  el lanzador $1 ha terminado" >> $L
