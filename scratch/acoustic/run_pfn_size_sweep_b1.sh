#!/usr/bin/env bash
# PFN encoder size sweep on B1, same recipe and sizes as run_pfn_size_sweep.sh (B2):
# set encoder, centroid frame, rot aug, var floor 0.01 m^2, bernoulli sensor dropout,
# calibration context, 200 epochs. The existing CL_B_pfnset predates the floor fix and
# dropout, so the width 256 / 6 layer reference is retrained here. No second seed (the
# B2 sweep measures seed noise). Waits for the B2 sweep. Two runs at a time; a run that
# fails (GPU memory) is retried alone at the end. Width 384 goes last.
cd /home/fer/tesis/sensor-agnostic-underwater-localization
PY=/home/fer/miniconda3/envs/torch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
DD=data/pfn10kd/hermite_distinct/processed/geometry_split
until grep -q "=== DONE" scratch/acoustic/run_pfn_size_sweep_out.log 2>/dev/null; do sleep 60; done
FAILED=scratch/acoustic/run_pfn_size_sweep_b1_failed.txt; : > $FAILED
train() {   # exp_name, extra overrides...
  name=$1; shift
  $PY scripts/train/train_np_geometry.py data.data_dir=$DD \
      data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
      model=spatial_pfn_set model.readout=joint data.rot_aug=true \
      data.context.per_sample=true data.context.calibration=true data.sensor_drop.mode=bernoulli \
      training.epochs=200 training.lr=1e-4 wandb.enabled=false exp_name=$name seed=0 "$@" \
      > scratch/acoustic/${name}_s0.log 2>&1
  rc=$?; echo "$name done $(date '+%F %T') (exit $rc)"
  [ $rc -ne 0 ] && echo "$name $*" >> $FAILED
  return 0
}
echo "start $(date '+%F %T')"
train CL_B_pfnset_d128_l6 model.spatial.set_d_model=128 &
train CL_B_pfnset_d192_l6 model.spatial.set_d_model=192 &
wait
train CL_B_pfnset_d256_l3 model.spatial.set_layers=3 &
train CL_B_pfnset_d256_l6 &
wait
while read -r name args; do echo "retry $name"; train $name $args; done < <(cat $FAILED; : > $FAILED)
train CL_B_pfnset_d384_l6 model.spatial.set_d_model=384
ck() { ls -d output/hydra/training/*/$1/seed_0 2>/dev/null | tail -1; }
M=""
for r in d128_l6 d192_l6 d256_l3 d256_l6 d384_l6; do
  d=$(ck CL_B_pfnset_$r); [ -n "$d" ] && [ -f $d/best.pt ] && M="$M $r=$d/best.pt"
done
for k in 0 3 5 7; do
  $PY scripts/eval/eval_calibration.py --data-dir $DD --models $M --drop $k \
      --out-dir output/eval_paper/b1_pfn_size_sweep_drop$k > scratch/acoustic/eval_b1_sweep_drop$k.log 2>&1
done
echo "=== DONE $(date '+%F %T') ==="
