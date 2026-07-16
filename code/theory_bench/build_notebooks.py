"""Сборка демонстрационных ноутбуков из сохранённых результатов экспериментов.

Запуск: python -m theory_bench.build_notebooks  (из каталога code/)
Создаёт notebooks/01..03 и исполняет их (все графики рендерятся из results/*.json,
GPU не требуется).
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

PRELUDE = '''import json, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

RES = "../theory_bench/results"
FIGS = "../theory_bench/figs"
pd.set_option("display.precision", 4)

def show(name, width=900):
    display(Image(filename=os.path.join(FIGS, name), width=width))
'''


def nb(cells, path, title):
    n = nbf.v4.new_notebook()
    n.cells = []
    for kind, src in cells:
        if kind == "md":
            n.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            n.cells.append(nbf.v4.new_code_cell(src))
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, path)
    print("built", path)


# ================================================================ 01: ярус A
cells1 = [
("md", """# Ярус A: проверка теории на малых задачах

Демонстрация экспериментов, проверяющих Теоремы 1–4 и леммы диплома на задачах
с точно вычислимыми решениями (float64, полный градиент, точные $w^*$ и $\\widetilde w^*$ методом Ньютона).

Все ячейки читают **сохранённые результаты** (`theory_bench/results/*.json`) — GPU не нужен.
Перезапуск любого эксперимента: `python -m theory_bench.experiments <a0|a1|...|s2> --device cuda:0`."""),
("code", PRELUDE),
("md", """## A0. Юнит-тест стенда

AdamW на диагональной квадратичной задаче: после стабилизации $\\hat D$ траектория должна
совпасть с аналитическим решением изменённой задачи $\\widetilde w^{*,i} = b_i/(a_i + \\lambda d^i)$,
при этом градиент **исходной** задачи остаётся большим."""),
("code", '''a0 = json.load(open(f"{RES}/a0.json"))
print(f"относительная ошибка до аналитического w̃*: {a0['rel_err_to_analytic']:.2e}")
print(f"||∇F̃||² = {a0['gFt2_final']:.2e},  ||∇F||² = {a0['gF2_final']:.3f}")
print(f"отношение критериев: {a0['ratio_gF2_gFt2']:.2e}")'''),
("md", """## A1. Главная иллюстрация: какой критерий убывает

У методов с затуханием весов (W) убывает $\\|\\nabla \\widetilde F_t\\|^2$, а $\\|\\nabla F\\|^2$
выходит на плато; у L2/WH — зеркально. Линия — медиана, заливка — разброс по 5 запускам."""),
("code", 'show("fig_main_adam.png")\nshow("fig_main_oasis.png")'),
("code", '''a1 = json.load(open(f"{RES}/a1.json"))
rows = []
for k, v in a1.items():
    task, upd, mode = k.split("|")
    rows.append(dict(task=task, update=upd, mode=mode, eta=v["eta"],
                     gF2_final=v["gF2_final"], gFt2_final=v["gFt2_final"]))
pd.DataFrame(rows).pivot_table(index=["task", "update"], columns="mode",
                               values=["gF2_final", "gFt2_final"], aggfunc="first")'''),
("md", """## A4. Лемма о нижней оценке расстояния между решениями

$L_F\\,\\|\\widetilde w^*_t - w^*\\| \\ge \\|(I-D_t)\\,\\nabla r(\\widetilde w^*_t)\\|$ — проверка на 27
конфигурациях с точными решениями (Ньютон). Все точки обязаны лежать выше диагонали."""),
("code", '''a4 = json.load(open(f"{RES}/a4.json"))
pts = a4["points"]
names = {"diagquad10": "квадратичная, $\\\\kappa=10$", "diagquad1e3": "квадратичная, $\\\\kappa=10^3$", "mushrooms": "mushrooms"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for t, mk in (("diagquad10", "o"), ("diagquad1e3", "s"), ("mushrooms", "^")):
    p = [q for q in pts if q["task"] == t]
    axes[0].loglog([q["RHS"] for q in p], [q["LHS"] for q in p], mk, label=names[t])
    axes[1].semilogx([q["lam"] for q in p], [q["tight"] for q in p], mk + "-", label=names[t])
lims = axes[0].get_xlim()
axes[0].loglog(lims, lims, "k--", alpha=0.5, label="$y=x$")
axes[0].set_xlabel(r"$\\|(I-D_t)\\,\\nabla r(\\widetilde w^*_t)\\|$"); axes[0].set_ylabel(r"$L_F\\,\\|\\widetilde w^*_t-w^*\\|$")
axes[1].set_xlabel(r"$\\lambda$"); axes[1].set_ylabel("отношение правой части к левой")
for a in axes: a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("нарушений неравенства:", len(a4["violations"]), "из", len(pts))'''),
("md", """## A5. Механизм адаптивной регуляризации

Элементы $d^i$ стационарной матрицы коррелируют с частотой признаков — weight decay
почти не штрафует редкие координаты."""),
("code", '''show("a5_dinf.png")
a5 = json.load(open(f"{RES}/a5.json"))
print("Spearman(d, частота):", round(a5["spearman_d_vs_freq"]["corr"], 3))
print("средний |w| на редких признаках: AdamW", round(a5["mean_abs_w_rare"]["adamw"], 2),
      "vs AdamL2", round(a5["mean_abs_w_rare"]["adaml2"], 2))'''),
("md", """## A3. Теорема 2: линейная скорость к $\\widetilde w^*_t$

Расстояние до движущейся цели убывает линейно; теоретическая скорость — корректная
(консервативная) гарантия: эмпирическая скорость всюду выше."""),
("code", '''show("a3_linear_rate.png")
a3 = json.load(open(f"{RES}/a3.json"))
pd.DataFrame([dict(alpha=r["cfg"]["alpha"], Gamma=r["cfg"]["gamma"], eta=r["eta"], tag=r["tag"],
                   rate_emp=r["rate_emp_per_iter"], rate_theory=r["rate_theory_per_iter"],
                   ratio=r["ratio_emp_over_theory"], R2_final=r["R2_final"]) for r in a3["runs"]])'''),
("md", """## A2. Лемма об эволюции $D_t$ (стохастический режим)

Прямое подтверждение: $\\max_t\\|\\hat D_{t+1}-\\hat D_t\\|_\\infty \\propto (1-\\beta)$,
граница леммы не нарушается. Уровень плато $\\|\\nabla\\widetilde F\\|^2$ при этом задаётся
шумовым членом (дрейфовый член — оценка худшего случая)."""),
("code", '''a2 = json.load(open(f"{RES}/a2.json"))
rr = [r for r in a2["runs"] if not r.get("is_baseline")]
fig, ax = plt.subplots(figsize=(6, 4))
for upd, mk in (("squares", "o"), ("linear", "s")):
    xs = [1 - r["beta"] for r in rr if r["update"] == upd]
    ys = [r["dD_max"] for r in rr if r["update"] == upd]
    bs = [r["rho_beta_bound"] for r in rr if r["update"] == upd]
    ax.loglog(xs, ys, mk + "-", label=f"измерено ({upd})")
    ax.loglog(xs, bs, mk + "--", alpha=0.5, label=f"граница леммы ({upd})")
ax.set_xlabel(r"$1-\\beta$"); ax.set_ylabel(r"$\\max_t\\|\\hat D_{t+1}-\\hat D_t\\|_\\infty$")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()'''),
("md", """## A7. Мосты к практическому Adam и стабилизация $D_t$

Замена срезки $\\alpha$ на практическую добавку $\\epsilon$, включение $\\beta_1=0.9$ и
bias-correction не меняют картину; поэлементного предела $D_t$ нет — стабилизируется масштаб."""),
("code", '''a7 = json.load(open(f"{RES}/a7.json"))
print("(a) clamp(α) vs +ε: падение ||∇F̃||², порядков:")
for k, v in a7["a_alpha_vs_eps"].items():
    print(f"   {k:12s} {v['gFt2_drop_orders']:.1f}")
print("(b) β₁ / bias-correction: финальные ||∇F̃||²:")
for k, v in a7["b_beta1_bias_correction"].items():
    print(f"   {k:22s} {v['gFt2_final']:.2e}")
show("a7_dstab.png", width=600)'''),
("md", """## S1b, S2. Шумовые члены Теорем 3 и 4

Плато $\\mathbb E\\|\\nabla\\widetilde F\\|^2 \\propto \\eta\\sigma^2$ (Т3, наклоны 1.000) и
шумовой шар $\\mathbb E\\|w-\\widetilde w^*\\|^2_D \\propto \\eta\\sigma^2$ под границей Т4 (наклон 1.03)."""),
("code", '''for name, fits_key in (("s1b", "fits"), ("s2", "fits")):
    d = json.load(open(f"{RES}/{name}.json"))
    print(name, json.dumps(d[fits_key], indent=1))
show("s1b_eta_scaling.png", width=600)
show("s2_noise_ball.png", width=600)'''),
]

# ================================================================ 02: ярус B
cells2 = [
("md", """# Ярус B: ResNet-18 и ViT-Tiny на CIFAR-10/100

Сетка $(\\eta, \\lambda)$, критерии на всей сети, чувствительность к $\\beta_2$
и обобщение при равном train loss. Все ячейки читают сохранённые результаты
(`theory_bench/results/dl/*.json`).

Перезапуск: `bash theory_bench/run_b1_grid.sh <gpu>`, `bash theory_bench/run_b4_tune.sh <gpu>`."""),
("code", PRELUDE + 'DL = f"{RES}/dl"'),
("md", """## B1. Heatmap $(\\eta, \\lambda)$: развязка у AdamW

90 прогонов ResNet-18/CIFAR-10. У AdamW качество нечувствительно к $\\lambda$ на пяти
порядках; у AdamL2 оптимум диагонален; AdamWH обрушивается вдоль $\\eta\\lambda \\approx \\mathrm{const}$."""),
("code", '''s = json.load(open(f"{DL}/b1_summary.json"))
lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]; wds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
names = {"l2": "AdamL2", "w": "AdamW", "wh": "AdamWH"}
acc = {m: np.array(s["acc_grid"][m]) for m in names}
vmax = max(np.nanmax(a) for a in acc.values())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for k, m in enumerate(names):
    ax = axes[k]
    im = ax.imshow(acc[m], origin="lower", aspect="auto", vmin=0.80, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(wds))); ax.set_xticklabels([f"{w:g}" for w in wds], fontsize=8)
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f"{l:g}" for l in lrs], fontsize=8)
    ax.set_xlabel(r"$\\lambda$");  ax.set_title(names[m])
    if k == 0: ax.set_ylabel(r"$\\eta$")
    for i in range(len(lrs)):
        for j in range(len(wds)):
            ax.text(j, i, f"{100*acc[m][i,j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if acc[m][i,j] < 0.5*(0.80+vmax) else "black")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="точность на тесте")
plt.show()
print("лучшие:", {names[m]: s["best"][m] for m in names})'''),
("md", """## Критерии на всей сети (лучший AdamW, 3 сида)

$\\|\\nabla F\\|^2$ выходит на плато, $\\|\\nabla \\widetilde F_t\\|^2$ убывает вместе с $\\|\\nabla f\\|^2$."""),
("code", '''runs = []
for sd in (0, 1, 2):
    d = json.load(open(f"{DL}/b1_w_lr0.01_wd0.1_s{sd}.json"))
    eps = sorted(d["diag"], key=int)
    runs.append(([int(e) for e in eps],
                 [d["diag"][e]["global"]["gF2"] for e in eps],
                 [d["diag"][e]["global"]["gFt2"] for e in eps]))
x = runs[0][0]
fig, ax = plt.subplots(figsize=(6.5, 4))
for idx, color, lbl in ((1, "tab:red", r"$\\|\\nabla F\\|^2$"), (2, "tab:blue", r"$\\|\\nabla \\widetilde F_t\\|^2$")):
    arr = np.array([r[idx] for r in runs])
    ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.2, lw=0)
    ax.semilogy(x, np.median(arr, 0), color=color, label=lbl)
ax.set_xlabel("эпоха"); ax.set_ylabel("квадрат нормы (пробный батч)")
ax.grid(alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()
print("финальное отношение критериев по сидам:", [f"{r[1][-1]/r[2][-1]:.0f}" for r in runs])'''),
("md", "## B2. Плато на сети не зависит от $\\beta_2$ (шумовой член доминирует)"),
("code", '''rows = []
for f in sorted(glob.glob(f"{DL}/b2beta*_w_*.json")):
    d = json.load(open(f))
    eps = sorted(d["diag"], key=int)
    rows.append(dict(beta2=d["beta2"],
                     plateau_gFt2=float(np.mean([d["diag"][e]["global"]["gFt2"] for e in eps[-3:]])),
                     gF2=float(np.mean([d["diag"][e]["global"]["gF2"] for e in eps[-3:]])),
                     best_acc=d["best_acc"]))
pd.DataFrame(rows).sort_values("beta2")'''),
("md", """## B4. CIFAR-100: итоговая точность и срезы «равный train loss»

При равной подгонке обобщение методов близко; преимущество AdamW — прежде всего
оптимизационное (AdamL2 на ViT не достигает train loss 2.0)."""),
("code", '''b4 = json.load(open(f"{DL}/b4_final.json"))
tbl = pd.DataFrame({k: dict(zip(["mean", "std"], v)) for k, v in b4["table"].items()}).T
display(tbl)
print("\\nсрезы (медианная точность на тесте при первом достижении train loss ≤ L):")
for k, r in b4["slices"].items():
    print(f"  {k:15s}:", {t: (round(v, 1) if v else None) for t, v in r.items()})'''),
]

# ================================================================ 03: ярус C
cells3 = [
("md", """# Ярус C: GPT-2 124M (FineWeb-Edu) и GLUE (RoBERTa-base)

Масштабирование эффектов на языковые модели: предобучение, рестарты с чекпойнтов,
послойная диагностика, дообучение на GLUE. Все ячейки читают сохранённые результаты.

Перезапуск: `python -m theory_bench.gpt_bench --mode w --lr 3e-4 --wd 0.01 --tokens 1.5e9 --tag main`."""),
("code", PRELUDE + 'GPT = f"{RES}/gpt"\nGLUE = f"{RES}/glue"'),
("md", "## Пилотная сетка (100M токенов): развязка AdamW, $\\lambda$-чувствительность AdamL2, расходимость AdamWH"),
("code", '''rows = []
for f in sorted(glob.glob(f"{GPT}/pilot_*.json")):
    d = json.load(open(f))
    rows.append(dict(mode=d["mode"], lr=d["lr"], wd=d["wd"], final_val=d["final_val"]))
pd.DataFrame(rows).sort_values(["mode", "lr", "wd"])'''),
("md", "## Основные запуски (1.5B токенов, 2 сида на метод)"),
("code", '''fig, ax = plt.subplots(figsize=(6.5, 4.2))
for mode, color, name in (("w", "tab:red", "AdamW"), ("l2", "tab:blue", "AdamL2")):
    cs = []
    for sd in (0, 1):
        d = json.load(open(f"{GPT}/main_{mode}_lr0.0003_wd0.01_s{sd}.json"))
        cs.append((np.array(d["val_at_tokens"]) / 1e9, np.array(d["val_loss"])))
    n = min(len(c[0]) for c in cs); x = cs[0][0][:n]
    ys = np.array([c[1][:n] for c in cs])
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.25, lw=0)
    ax.plot(x, ys.mean(0), color=color, label=name)
ax.set_xlabel("токены, млрд"); ax.set_ylabel("val loss"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
for mode in ("w", "l2"):
    vals = [json.load(open(f"{GPT}/main_{mode}_lr0.0003_wd0.01_s{s}.json"))["final_val"] for s in (0, 1)]
    print(mode, "final val:", [round(v, 4) for v in vals])'''),
("md", """## Послойная диагностика: авто-«no-decay»

Отношение $\\|\\nabla F\\|^2/\\|\\nabla\\widetilde F_t\\|^2$ в конце обучения: LayerNorm 450–1400,
эмбеддинги ≈14, attention/MLP 1.5–3 — адаптивная регуляризация сама отключает штраф там,
где градиенты малы."""),
("code", '''la = json.load(open(f"{GPT}/final_layer_analysis.json"))["w"]
order = ["emb", "low_attn", "mid_attn", "high_attn", "low_mlp", "mid_mlp", "high_mlp",
         "low_ln", "mid_ln", "high_ln", "final"]
labels = ["emb", "attn 0-3", "attn 4-7", "attn 8-11", "mlp 0-3", "mlp 4-7", "mlp 8-11",
          "LN 0-3", "LN 4-7", "LN 8-11", "LN fin"]
vals = [la[g]["ratio"] for g in order]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(range(len(order)), vals, color=["tab:purple"] + ["tab:orange"]*3 + ["tab:green"]*3 + ["tab:red"]*4)
ax.set_yscale("log"); ax.axhline(1, color="gray", ls=":", lw=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel(r"$\\|\\nabla F\\|^2/\\|\\nabla\\widetilde F_t\\|^2$")
ax.grid(alpha=0.3, axis="y"); plt.tight_layout(); plt.show()'''),
("md", "## Рестарты с чекпойнта: эффект — свойство метода, а не траектории"),
("code", '''fig, ax = plt.subplots(figsize=(6.2, 4))
for i, sd in enumerate((101, 102, 103)):
    d = json.load(open(f"{GPT}/branch_w_seed{sd}.json"))
    steps = sorted(d["diag"], key=int)
    x = np.array([int(s) for s in steps]) * 0.24576
    ax.semilogy(x, [d["diag"][s]["global"]["gF2"] for s in steps], color="tab:red",
                alpha=0.5 + 0.2*i, label=r"$\\|\\nabla F\\|^2$" if i == 0 else None)
    ax.semilogy(x, [d["diag"][s]["global"]["gFt2"] for s in steps], color="tab:blue",
                alpha=0.5 + 0.2*i, label=r"$\\|\\nabla \\widetilde F_t\\|^2$" if i == 0 else None)
    last = steps[-1]; g = d["diag"][last]["global"]
    print(f"ветка {sd}: val {d['val_loss'][0]:.4f} -> {d['val_loss'][-1]:.4f}, отношение критериев {g['gF2']/g['gFt2']:.2f}")
ax.set_xlabel("токены после рестарта, млн"); ax.set_ylabel("квадрат нормы (вся сеть)")
ax.grid(alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()'''),
("md", """## Спасение AdamWH: правило $\\eta\\lambda/\\epsilon$ работает в обе стороны

Поднятие $\\epsilon \\ge \\eta\\lambda$ устраняет расходимость, но сплющивает предобуславливатель —
метод деградирует. Рабочей точки на претрейне LLM нет."""),
("code", '''for f in sorted(glob.glob(f"{GPT}/rescue_*.json")):
    d = json.load(open(f))
    print(os.path.basename(f), "-> final val:", round(d["final_val"], 3))'''),
("md", "## GLUE (RoBERTa-base): итоговая таблица"),
("code", '''ft = json.load(open(f"{GLUE}/final_table.json"))
tbl = {}
for task, row in ft.items():
    for mode, (m, s) in row.items():
        tbl.setdefault({"l2": "AdamL2", "w": "AdamW", "wh": "AdamWH"}[mode], {})[task] = f"{100*m:.1f} ± {100*s:.1f}"
# SST-2 из отдельных прогонов
for mode, name in (("l2", "AdamL2"), ("w", "AdamW"), ("wh", "AdamWH")):
    cand = {}
    for f in glob.glob(f"{GLUE}/sst2_{mode}_*_s0.json"):
        d = json.load(open(f)); cand[d["lr"]] = d["best_metric"] or -1
    lr = max(cand, key=cand.get)
    vals = []
    for s in (0, 1, 2):
        p = f"{GLUE}/sst2_{mode}_lr{lr:g}_wd0.01_s{s}.json"
        if os.path.exists(p):
            vals.append(json.load(open(p))["best_metric"])
    v = np.array(vals) * 100
    tbl[name]["sst2"] = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
pd.DataFrame(tbl).T[["rte", "mrpc", "cola", "stsb", "sst2"]]'''),
]

nb(cells1, os.path.join(NB_DIR, "01_theory_small_scale.ipynb"), "A")
nb(cells2, os.path.join(NB_DIR, "02_cifar_resnet_vit.ipynb"), "B")
nb(cells3, os.path.join(NB_DIR, "03_gpt2_glue.ipynb"), "C")

# ---- исполнение (рендер вывода)
from nbclient import NotebookClient
for name in ("01_theory_small_scale", "02_cifar_resnet_vit", "03_gpt2_glue"):
    path = os.path.join(NB_DIR, name + ".ipynb")
    n = nbf.read(path, as_version=4)
    client = NotebookClient(n, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": NB_DIR}})
    client.execute()
    nbf.write(n, path)
    print("executed", name)
print("DONE")
