#!/bin/bash
# usage example: bash ucr_ablaton.sh GPU_NUMBER DATASET

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${1:-0}

datasets=(
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

mkdir -p ./results/SREA_ablation_hyperpar/LOGs/$2/

dt=$(date '+%Y-%m-%dT%H:%M:%S')
logname=./results/SREA_ablation_hyperpar/LOGs/$2/$dt.txt
touch $logname
echo $dt LOGFILE: $logname
python src/SREA_ablation_hyperpar.py --dataset $2 --correct True \
  --delta_track 5 --init 1 10 20 40 --delta_init 0 5 10 15 25 --delta_end 10 20 30 \
  --label_noise 0 --ni 0.15 0.30 0.45 0.6 --abg 1 1 1  2>&1 | tee $logname

dt=$(date '+%Y-%m-%dT%H:%M:%S')
logname=./results/SREA_ablation_hyperpar/LOGs/$2/$dt.txt
touch $logname
echo $dt LOGFILE: $logname
python src/SREA_ablation_hyperpar.py --dataset $2 --correct True \
  --delta_track 5 --init 1 10 20 40 --delta_init 0 5 10 15 25 --delta_end 10 20 30 \
  --label_noise 1 --ni 0.1 0.2 0.3 0.4 --abg 1 1 1  2>&1 | tee $logname
