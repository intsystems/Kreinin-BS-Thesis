#!/bin/bash
# Возобновление основных прогонов C2 после сбоя/перезагрузки.
# Каждый прогон сам подхватит свой чекпойнт (ckpt_main_*.pt, сохраняется каждые 500 шагов)
# и продолжит с места остановки. Уже завершённые прогоны перезапускать не нужно
# (у них есть main_*_s*.json) — скрипт их пропускает.
cd "$(dirname "$0")/.."
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
launch() { # gpu mode seed
  local gpu=$1 mode=$2 seed=$3
  local out="theory_bench/results/gpt/main_${mode}_lr0.0003_wd0.01_s${seed}.json"
  [ -f "$out" ] && { echo "skip $out (готов)"; return; }
  pgrep -f "gpt_bench --mode $mode .*--seed $seed --tag main" > /dev/null && { echo "уже идёт: $mode s$seed"; return; }
  echo "resume: gpu$gpu $mode seed$seed"
  setsid nohup env CUDA_VISIBLE_DEVICES=$gpu $PY -m theory_bench.gpt_bench \
    --mode "$mode" --lr 3e-4 --wd 0.01 --tokens 1.5e9 --seed "$seed" --tag main --device cuda:0 \
    > "theory_bench/results/gpt/logs/main_${mode}_s${seed}_resume.log" 2>&1 < /dev/null &
}
launch 0 l2 1
launch 1 w 0
launch 2 l2 0
launch 3 w 1
echo "done"
