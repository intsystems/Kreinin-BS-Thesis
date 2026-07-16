#!/bin/bash
# B4 тюнинг (seed 0): CIFAR-100, {resnet18 x 60 эпох, vit x 100 эпох} x {l2,w,wh,sgd} x сетка 3x3.
# Использование: bash run_b4_tune.sh <gpu 0..3> — карта берёт конфигурации с i % 4 == gpu.
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."
mkdir -p theory_bench/results/dl/logs

i=0
run_one() { # model epochs mode lr wd
  local model=$1 epochs=$2 mode=$3 lr=$4 wd=$5
  if [ $((i % 4)) -eq "$GPU" ]; then
    local out="theory_bench/results/dl/b4_${model}_cifar100_${mode}_lr${lr}_wd${wd}_s0.json"
    if [ -f "$out" ]; then echo "skip $out"; else
      echo "[gpu$GPU] b4 $model $mode lr=$lr wd=$wd"
      CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.dl_bench \
        --mode "$mode" --lr "$lr" --wd "$wd" --seed 0 --epochs "$epochs" \
        --dataset cifar100 --model "$model" --tag b4 --device cuda:0 \
        >> "theory_bench/results/dl/logs/b4_gpu${GPU}.log" 2>&1 \
        || echo "FAILED: $model $mode $lr $wd" >> "theory_bench/results/dl/logs/b4_gpu${GPU}.log"
    fi
  fi
  i=$((i + 1))
}

for model_epochs in "resnet18 60" "vit 100"; do
  set -- $model_epochs; model=$1; epochs=$2
  for mode in l2 w wh; do
    for lr in 3e-4 1e-3 3e-3; do
      for wd in 1e-3 1e-2 1e-1; do
        run_one "$model" "$epochs" "$mode" "$lr" "$wd"
      done
    done
  done
  for lr in 0.03 0.1 0.3; do
    for wd in 1e-4 5e-4 5e-3; do
      run_one "$model" "$epochs" sgd "$lr" "$wd"
    done
  done
done
echo "B4 tune gpu$GPU DONE"
