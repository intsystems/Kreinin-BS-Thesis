#!/bin/bash
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."
i=0
while read -r model mode lr wd; do
  epochs=60; [ "$model" = "vit" ] && epochs=100
  for seed in 1 2 3 4; do
    if [ $((i % 4)) -eq "$GPU" ]; then
      out="theory_bench/results/dl/b4_${model}_cifar100_${mode}_lr${lr}_wd${wd}_s${seed}.json"
      if [ -f "$out" ]; then echo "skip $out"; else
        CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.dl_bench \
          --mode "$mode" --lr "$lr" --wd "$wd" --seed "$seed" --epochs "$epochs" \
          --dataset cifar100 --model "$model" --tag b4 --device cuda:0 \
          >> "theory_bench/results/dl/logs/b4f_gpu${GPU}.log" 2>&1 \
          || echo "FAILED $model $mode s$seed" >> "theory_bench/results/dl/logs/b4f_gpu${GPU}.log"
      fi
    fi
    i=$((i + 1))
  done
done < <($PY -c "
import json
cfg = json.load(open('theory_bench/results/dl/b4_best.json'))
for k, (lr, wd) in cfg.items():
    m, mo = k.split('|')
    print(m, mo, f'{lr:g}', f'{wd:g}')")
echo "B4 finals gpu$GPU DONE"
