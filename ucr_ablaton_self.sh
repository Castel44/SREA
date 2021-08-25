#!/bin/bash
# example ucr_ablaton.sh GPU_NUMBER DATASET

available_datasets=(
  'CBF'
  'Trace'
  'Plane'
  'Symbols'
  'OSULeaf'
  'FaceFour'
  'ArrowHead'
  'MelbournePedestrian'
  'Epilepsy'
  'NATOPS'
  'PenDigits'
)


export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${1:-0}
dataset=( "${@:2}" )

len="${#dataset[@]}"

for ((i=0; i<$len; i++))
do
  dt=$(date '+%Y-%m-%dT%H:%M:%S')
  echo $dt - 'Symmetric Noise' - "${dataset[$i]}": [$i$total]
  python src/SREA_ablation_hyperpar.py --dataset "${dataset[$i]}" \
   --correct False True --n_runs 10 --process 2 \
  --delta_track 5 --init 10 --delta_init 25 --delta_end 30 \
  --label_noise 0 --ni 0.0 0.15 0.30 0.45 0.6 0.75

  echo $dt - 'Asymmetric Noise' - "${dataset[$i]}": [$i$total]
  python src/SREA_ablation_hyperpar.py --dataset "${dataset[$i]}" \
   --correct False True --n_runs 10 --process 2 \
  --delta_track 5 --init 10 --delta_init 25 --delta_end 30 \
  --label_noise 1 --ni 0.0 0.1 0.2 0.3 0.4 0.5
done



