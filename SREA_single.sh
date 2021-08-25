#! /bin/bash
# single experiment with SREA
# usage: bash SREA_single.sh DATASET GPU_NUMBER NOISE_TYPE NOISE_RATIO N_RUNS

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
export CUDA_VISIBLE_DEVICES=${2:-0}


python src/SREA_single_experiment.py --dataset $1