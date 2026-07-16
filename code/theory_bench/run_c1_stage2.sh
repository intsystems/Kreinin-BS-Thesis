#!/bin/bash
# C1 стадия 2: сиды 1-4 на лучших (lr, wd) из best_cfg.json => 48 прогонов.
set -u
GPU=$1
PY=/home/kreinin.mv/venvs/thesis-bench/bin/python
cd "$(dirname "$0")/.."
mkdir -p theory_bench/results/glue/logs

$PY - <<'PYEOF' > /tmp/c1_stage2_configs.txt
import json
best = json.load(open('theory_bench/results/glue/best_cfg.json'))
for key, (lr, wd) in best.items():
    task, mode = key.split('|')
    for seed in (1, 2, 3, 4):
        print(task, mode, f"{lr:g}", f"{wd:g}", seed)
PYEOF

while read -r task mode lr wd seed; do
  out="theory_bench/results/glue/${task}_${mode}_lr${lr}_wd${wd}_s${seed}.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "[gpu$GPU] $task $mode lr=$lr wd=$wd seed=$seed"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m theory_bench.glue_bench \
    --task "$task" --mode "$mode" --lr "$lr" --wd "$wd" --seed "$seed" --device cuda:0 \
    >> "theory_bench/results/glue/logs/c1s2_gpu${GPU}.log" 2>&1 \
    || echo "FAILED: $task $mode $lr $wd s$seed" >> "theory_bench/results/glue/logs/c1s2_gpu${GPU}.log"
done < /tmp/c1_stage2_configs.txt
echo "C1 stage2 gpu$GPU DONE"
