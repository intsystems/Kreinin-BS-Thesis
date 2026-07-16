"""Генерация фигур для раздела экспериментов диплома.

fig_main_adam.png / fig_main_oasis.png — главная иллюстрация: mushrooms, три режима
регуляризации, два критерия. Остальные фигуры берутся из figs/ как есть.
Запуск: python -m theory_bench.thesis_figs --device cuda:0
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .problems import LogReg, L2Reg
from .experiments import run_traj, FIGS, MUSHROOMS, COLORS, LBL

SKIP = 10  # пропускаем первые точки лога (начальный выброс при D ~ alpha)


def main(device):
    lam = 0.04
    reg = L2Reg(lam)
    pb = LogReg(MUSHROOMS, device=device)
    # eta подобраны в эксперименте A1 (results/a1.json)
    cfg = {
        ("squares", "l2"): (0.003, 1e-8), ("squares", "wh"): (0.01, 1e-2), ("squares", "w"): (0.03, 1e-8),
        ("linear", "l2"): (0.1, 1e-8), ("linear", "wh"): (0.1, 1e-2), ("linear", "w"): (0.1, 1e-8),
    }
    T = 20000
    for update, fam, fname in (("squares", "Adam", "fig_main_adam"),
                               ("linear", "OASIS", "fig_main_oasis")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for mode in ("l2", "wh", "w"):
            eta, alpha = cfg[(update, mode)]
            gF2s, gFt2s = [], []
            for s in range(5):
                _, _, lg = run_traj(pb, reg, mode, update, eta, 0.999, alpha, 1e3,
                                    T, seed=s, device=device, log_every=20)
                gF2s.append(lg["gF2"])
                gFt2s.append(lg["gFt2"])
            t = np.array(lg["t"][SKIP:])
            a0 = np.array(gF2s)[:, SKIP:]
            a1 = np.array(gFt2s)[:, SKIP:]
            # медиана + межсидовый разброс (min-max) — устойчивость к рестартам
            for ax, arr in ((axes[0], a0), (axes[1], a1)):
                ax.fill_between(t, arr.min(0), arr.max(0), color=COLORS[mode], alpha=0.18, lw=0)
                ax.semilogy(t, np.median(arr, 0), color=COLORS[mode], label=fam + LBL[mode])
        axes[0].set_ylabel(r"$\|\nabla F(w_t)\|^2$")
        axes[1].set_ylabel(r"$\|\nabla \widetilde F_t(w_t)\|^2$")
        for a in axes:
            a.set_xlabel("итерация")
            a.grid(alpha=0.3)
            a.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, fname + ".png"), dpi=150)
        plt.close(fig)
        print("saved", fname, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    main(args.device)
