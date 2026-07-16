#!/bin/bash
# Ярус B, эксперимент B1: heatmap (eta, lambda) для AdamL2/AdamW/AdamWH на ResNet-18/CIFAR-10.
# 3 режима x 5 lr x 6 wd = 90 прогонов по 50 эпох, seed 0.
# Использование: bash run_b1_grid.sh <gpu_id 0|1>  — карта с номером gpu берёт свою половину сетки.
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."   # -> code/
mkdir -p theory_bench/results/dl/logs

i=0
for mode in l2 w wh; do
  for lr in 1e-4 3e-4 1e-3 3e-3 1e-2; do
    for wd in 1e-5 1e-4 1e-3 1e-2 1e-1 1; do
      if [ $((i % 2)) -eq "$GPU" ]; then
        tagfile="theory_bench/results/dl/b1_${mode}_lr${lr}_wd${wd}_s0.json"
        if [ -f "$tagfile" ]; then
          echo "skip existing $tagfile"
        else
          echo "[gpu$GPU] run $i: mode=$mode lr=$lr wd=$wd"
          CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.dl_bench \
            --mode "$mode" --lr "$lr" --wd "$wd" --seed 0 --epochs 50 --tag b1 --device cuda:0 \
            >> "theory_bench/results/dl/logs/b1_gpu${GPU}.log" 2>&1 \
            || echo "FAILED: $mode lr=$lr wd=$wd" >> "theory_bench/results/dl/logs/b1_gpu${GPU}.log"
        fi
      fi
      i=$((i + 1))
    done
  done
done
echo "B1 grid gpu$GPU DONE"
