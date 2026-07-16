#!/bin/bash
# C2 пилот: 3 метода x LR {3e-4, 6e-4} x wd {0.01, 0.1} = 12 прогонов по 100M токенов.
# Использование: bash run_c2_pilot.sh <gpu 0|1> — карта берёт свою половину по чётности.
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."
mkdir -p theory_bench/results/gpt/logs

i=0
for mode in l2 w wh; do
  for lr in 3e-4 6e-4; do
    for wd in 0.01 0.1; do
      if [ $((i % 2)) -eq "$GPU" ]; then
        out="theory_bench/results/gpt/pilot_${mode}_lr${lr}_wd${wd}_s0.json"
        if [ -f "$out" ]; then
          echo "skip $out"
        else
          echo "[gpu$GPU] pilot: $mode lr=$lr wd=$wd"
          CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.gpt_bench \
            --mode "$mode" --lr "$lr" --wd "$wd" --tokens 1e8 --seed 0 --tag pilot --device cuda:0 \
            >> "theory_bench/results/gpt/logs/pilot_gpu${GPU}.log" 2>&1 \
            || echo "FAILED: $mode $lr $wd" >> "theory_bench/results/gpt/logs/pilot_gpu${GPU}.log"
        fi
      fi
      i=$((i + 1))
    done
  done
done
echo "C2 pilot gpu$GPU DONE"
