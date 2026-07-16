#!/bin/bash
# C1 стадия 1: LR-сетка. 4 задачи x 3 метода x 3 LR, seed 0 => 36 прогонов, деление по чётности.
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."
mkdir -p theory_bench/results/glue/logs

i=0
for task in rte mrpc cola stsb; do
  for mode in l2 w wh; do
    for lr in 1e-5 2e-5 3e-5; do
      if [ $((i % 2)) -eq "$GPU" ]; then
        out="theory_bench/results/glue/${task}_${mode}_lr${lr}_wd0.1_s0.json"
        if [ -f "$out" ]; then
          echo "skip $out"
        else
          echo "[gpu$GPU] $task $mode lr=$lr"
          CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.glue_bench \
            --task "$task" --mode "$mode" --lr "$lr" --wd 0.1 --seed 0 --device cuda:0 \
            >> "theory_bench/results/glue/logs/c1s1_gpu${GPU}.log" 2>&1 \
            || echo "FAILED: $task $mode $lr" >> "theory_bench/results/glue/logs/c1s1_gpu${GPU}.log"
        fi
      fi
      i=$((i + 1))
    done
  done
done
echo "C1 stage1 gpu$GPU DONE"
