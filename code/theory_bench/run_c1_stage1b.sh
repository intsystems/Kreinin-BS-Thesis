#!/bin/bash
# C1 стадия 1b: контроль lambda. Все методы wd=0.01 (36 прогонов) + wh wd=0.001 (12 прогонов).
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."
mkdir -p theory_bench/results/glue/logs

run() {
  local task=$1 mode=$2 lr=$3 wd=$4
  local out="theory_bench/results/glue/${task}_${mode}_lr${lr}_wd${wd}_s0.json"
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  echo "[gpu$GPU] $task $mode lr=$lr wd=$wd"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.glue_bench \
    --task "$task" --mode "$mode" --lr "$lr" --wd "$wd" --seed 0 --device cuda:0 \
    >> "theory_bench/results/glue/logs/c1s1b_gpu${GPU}.log" 2>&1 \
    || echo "FAILED: $task $mode $lr $wd" >> "theory_bench/results/glue/logs/c1s1b_gpu${GPU}.log"
}

for task in rte mrpc cola stsb; do
  for mode in l2 w wh; do
    for lr in 1e-5 2e-5 3e-5; do
      run "$task" "$mode" "$lr" 0.01
    done
  done
  for lr in 1e-5 2e-5 3e-5; do
    run "$task" wh "$lr" 0.001
  done
done
echo "C1 stage1b gpu$GPU DONE"
