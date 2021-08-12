#!/bin/bash
# example ucr_ablaton.sh GPU_NUMBER DATASET

export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=${1:-0}

all_datasets=(
  #'CBF'
  #'Trace'
  #'Plane'
  'Symbols'
  'OSULeaf'
  'FaceFour'
  'ArrowHead'
  'MelbournePedestrian'
  'Epilepsy'
  'NATOPS'
  #'PenDigits'
)

total="${#all_datasets[@]}"
start=`date +%s`


for ((i=0; i<$total; i++))
do
  echo $i/$total: "${all_datasets[$i]}"

  python src/ucr_labelnoise_SOTA.py --dataset "${all_datasets[$i]}" \
  --label_noise 1 --ni 0 0.1 0.2 0.3 0.4 0.5 \
  --n_runs 5

  cur=`date +%s`
  count=$((i+1))
  pd=$(( $count * 73 / $total ))
  runtime=$(( $cur-$start ))
  estremain=$(( ($runtime * $total / $count)-$runtime ))
  printf "%d.%d%% complete ($count of $total) - est %d:%0.2d remaining \n " $(( $count*100/$total )) $(( ($count*1000/$total)%10)) $(( $estremain/60 )) $(( $estremain%60 ))
done
printf "\ndone\n"


