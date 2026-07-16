# Код экспериментов

Все эксперименты диплома воспроизводятся стендом `theory_bench/`. Результаты прогонов
сохраняются в `theory_bench/results/**/*.json`, готовые графики — в `theory_bench/figs/`
(в диплом копируются в `docs/tex/pictures/`). JSON-результаты и графики версионируются
(≈7 МБ — доказательная база диплома); чекпойнты `*.pt`, каталоги `data/` и `logs/` — нет.

Ноутбуки в `notebooks/` рендерят все графики и таблицы диплома из сохранённых
результатов: выводы уже запечены в файлы, а для перезапуска ячеек GPU не нужен —
достаточно `results/` и `figs/`.

## Структура

```
code/
├── mushrooms.txt              # датасет LIBSVM для логистической регрессии
├── requirements.txt
├── notebooks/
│   ├── 01_theory_small_scale.ipynb   # ярус A: проверка теорем и лемм на малых задачах
│   ├── 02_cifar_resnet_vit.ipynb     # ярус B: ResNet/ViT на CIFAR-10/100
│   └── 03_gpt2_glue.ipynb            # ярус C: GPT-2 124M и GLUE
└── theory_bench/
    ├── problems.py            # задачи: квадратичные, логрегрессия, невыпуклая классификация
    ├── precond.py             # единый оптимизатор: Adam/OASIS × {L2, W, WH}, срезка, критерии
    ├── exact.py               # точные решения (Ньютон) и константы
    ├── experiments.py         # ярус A: a0–a7, s1, s1b, s2 (CLI)
    ├── thesis_figs.py         # главные фигуры диплома (mushrooms, полосы по сидам)
    ├── dl_bench.py            # ярус B: ResNet-18/ViT-Tiny, CIFAR-10/100, AdamL2/W/WH/SGD
    ├── glue_bench.py          # ярус C1: RoBERTa-base на GLUE
    ├── gpt_bench.py           # ярус C2: GPT-2 124M (nanoGPT-стиль, чекпойнты)
    ├── fineweb_prep.py        # подготовка данных FineWeb-Edu -> train.bin/val.bin
    ├── restart_branch.py      # ветвление обучения с чекпойнта (проверка рестартов)
    ├── build_notebooks.py     # (пере)сборка и исполнение ноутбуков из results/
    ├── run_b1_grid.sh         # B1: сетка (eta, lambda) на CIFAR-10
    ├── run_b4_tune.sh         # B4: тюнинг LR на CIFAR-100 (обе модели, 4 метода)
    ├── run_b4_finals.sh       # B4: финальные прогоны с 3 сидами
    ├── run_c1_stage1.sh       # GLUE: сетка LR (этап 1)
    ├── run_c1_stage1b.sh      # GLUE: сетка LR для WH (этап 1b)
    ├── run_c1_stage2.sh       # GLUE: финалы с сидами (этап 2)
    ├── run_c2_pilot.sh        # GPT-2: пилотная сетка на 100M токенов
    ├── resume_mains.sh        # возобновление прогонов GPT-2 с чекпойнтов
    ├── results/               # JSON-результаты всех прогонов (версионируются)
    └── figs/                  # готовые графики (версионируются)
```

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Юнит-тест стенда (сходимость AdamW к аналитическому решению изменённой задачи):

```bash
python -m theory_bench.experiments a0 --device cuda:0
```

## Воспроизведение по ярусам

Ярус A (малые задачи, ~30 GPU-мин суммарно):

```bash
for e in a0 a1 a2 a3 a4 a5 a6 a7 s1 s1b s2; do
  python -m theory_bench.experiments $e --device cuda:0
done
python -m theory_bench.thesis_figs --device cuda:0   # главные фигуры с полосами по сидам
```

Ярус B (CIFAR, ~30 GPU-ч): `bash theory_bench/run_b1_grid.sh <gpu>` (heatmap),
`bash theory_bench/run_b4_tune.sh <gpu>` + финалы с сидами (см. run_b4_finals.sh).

Ярус C (~60 GPU-ч): подготовка данных `python -m theory_bench.fineweb_prep --tokens 3.2e9`,
GLUE `bash theory_bench/run_c1_stage1.sh <gpu>`, GPT-2
`python -m theory_bench.gpt_bench --mode w --lr 3e-4 --wd 0.01 --tokens 1.5e9 --tag main`.
Прогоны GPT-2 сохраняют чекпойнты каждые 500 шагов и возобновляются тем же запуском.

## Соглашения стенда

- Критерий $\|\nabla \tilde F_t\|^2 = \|\nabla f + \hat D_t \nabla r\|^2$ всюду вычисляется
  с той же матрицей $\hat D_t$ (включая bias-correction и eps), что используется в шаге.
- Три режима регуляризации: `l2` (в градиент), `w` (decoupled), `wh` (масштабированное).
- В ярусе A матрица срезается с двух сторон: $\hat D = \mathrm{clip}(|D|, \alpha, \Gamma)$.
