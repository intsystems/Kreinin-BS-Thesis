"""Эксперименты яруса A (проверка теории) и стохастические S1/S2 (Теоремы 3-4).

Запуск:  python -m theory_bench.experiments <exp> [--device cuda:0]
где <exp> из {a0, a1, a2, a3, a4, a5, a6, s1, s2}.

Результаты: theory_bench/results/<exp>.json, картинки: theory_bench/figs/<exp>_*.png
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .problems import DiagQuadratic, Quadratic, LogReg, NonconvexClf, L2Reg, BoundedReg
from .precond import PrecondOptimizer, criteria
from .exact import newton_F, constants

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
MUSHROOMS = os.path.join(os.path.dirname(HERE), "mushrooms.txt")

COLORS = {"l2": "tab:blue", "wh": "tab:orange", "w": "tab:red"}
LBL = {"l2": "L2", "wh": "WH", "w": "W"}


def save_json(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
    print("saved", name + ".json", flush=True)


def w0_init(d, seed, device, scale=1.0):
    g = torch.Generator().manual_seed(10000 + seed)
    return (scale * torch.randn(d, generator=g, dtype=torch.float64)).to(device)


def run_traj(problem, reg, mode, update, eta, beta2, alpha, gamma, T, seed=0,
             device="cpu", stochastic=False, delayed=False, log_every=10,
             freeze_at=None, beta1=0.0, w0=None):
    """Универсальный прогон. Возвращает (w_T, D_hat_T, logs)."""
    w = w0_init(problem.d, seed, device) if w0 is None else w0.clone()
    opt = PrecondOptimizer(problem, reg, mode=mode, update=update, eta=eta,
                           beta1=beta1, beta2=beta2, alpha=alpha, gamma=gamma,
                           delayed=delayed, seed=seed, device=device)
    logs = {"t": [], "gF2": [], "gFt2": [], "dD": []}
    D_hat = None
    for t in range(T):
        if freeze_at is not None and t == freeze_at:
            opt.beta2 = 1.0
        w, info = opt.step(w, stochastic=stochastic)
        D_hat = info["D_hat"]
        if t % log_every == 0 or t == T - 1:
            gF2, gFt2 = criteria(problem, reg, w, D_hat)
            logs["t"].append(t)
            logs["gF2"].append(gF2)
            logs["gFt2"].append(gFt2)
            logs["dD"].append(info["dD_inf"] if info["dD_inf"] is not None else 0.0)
    return w, D_hat, logs


# ================================================================ A0: юнит-тест
def a0(device):
    """AdamW на диагональной квадратичной: после заморозки D и стабилизации шага
    w_T должен совпасть с аналитическим w~*(D) = b/(a + lam*D); grad F при этом НЕ мал."""
    pb = DiagQuadratic(d=100, kappa=1e3, device=device)
    lam = 0.1
    reg = L2Reg(lam)
    w = w0_init(pb.d, 0, device)
    opt = PrecondOptimizer(pb, reg, mode="w", update="squares", eta=0.05,
                           beta2=0.999, alpha=1e-8, gamma=1e3, device=device)
    D = None
    for t in range(20000):
        w, info = opt.step(w)
        D = info["D_hat"]
    # фаза 2: замораживаем D (beta2=1) и берём гарантированно устойчивый шаг
    opt.beta2 = 1.0
    opt.eta = 0.9 / float((pb.a / D + lam).max())
    for t in range(100000):
        w, info = opt.step(w)
        D = info["D_hat"]
    wt_analytic = pb.wstar_tilde(lam, D)
    rel = float((w - wt_analytic).norm() / wt_analytic.norm())
    gF2, gFt2 = criteria(pb, reg, w, D)
    ok = rel < 1e-8 and gF2 > 1e6 * max(gFt2, 1e-300)
    out = {"rel_err_to_analytic": rel, "gF2_final": gF2, "gFt2_final": gFt2,
           "ratio_gF2_gFt2": gF2 / max(gFt2, 1e-300), "eta_phase2": opt.eta,
           "PASS": bool(ok)}
    save_json("a0", out)
    print("A0", "PASS" if ok else "FAIL", out, flush=True)
    return ok


# ================================================================ A1: главная иллюстрация
def a1(device):
    tasks = [
        ("diagquad", DiagQuadratic(d=100, kappa=1e3, device=device), 0.1),
        ("quad", Quadratic(d=100, kappa=1e3, device=device), 0.1),
        ("mushrooms", LogReg(MUSHROOMS, device=device), 0.04),
    ]
    seeds = [0, 1, 2, 3, 4]
    curves, summary = {}, {}
    for tname, pb, lam in tasks:
        reg = L2Reg(lam)
        for update in ("squares", "linear"):
            # у Adam-семейства (squares) на жёстких квадратичных осцилляционный пол ~ (eta*L)^2:
            # для честного контраста критериев нужна сетка eta вниз и длиннее прогон
            if update == "squares" and tname != "mushrooms":
                etas, T = [1e-4, 3e-4, 1e-3, 3e-3], 100000
            else:
                etas, T = [3e-3, 1e-2, 3e-2, 1e-1], 20000
            for mode in ("l2", "wh", "w"):
                key = f"{tname}|{update}|{mode}"
                # WH неустойчив при eta*lam/alpha > 2 (регуляризатор проходит через D^-1),
                # поэтому для WH поднимаем нижнюю срезку
                alpha = 1e-2 if mode == "wh" else 1e-8
                best = None
                for eta in etas:
                    crit = "gFt2" if mode == "w" else "gF2"
                    _, D, lg = run_traj(pb, reg, mode, update, eta, 0.999, alpha, 1e3,
                                        T, seed=0, device=device, log_every=20)
                    final = lg[crit][-1]
                    if not np.isfinite(final):
                        continue
                    if best is None or final < best[0]:
                        best = (final, eta)
                eta = best[1] if best is not None else etas[0]
                gF2s, gFt2s = [], []
                for s in seeds:
                    _, D, lg = run_traj(pb, reg, mode, update, eta, 0.999, alpha, 1e3,
                                        T, seed=s, device=device, log_every=20)
                    gF2s.append(lg["gF2"])
                    gFt2s.append(lg["gFt2"])
                curves[key] = {
                    "t": lg["t"],
                    "gF2": np.median(np.array(gF2s), 0).tolist(),
                    "gFt2": np.median(np.array(gFt2s), 0).tolist(),
                    "eta": eta,
                }
                summary[key] = {"eta": eta, "gF2_final": curves[key]["gF2"][-1],
                                "gFt2_final": curves[key]["gFt2"][-1]}
                print("A1", key, "eta=", eta, summary[key], flush=True)
    save_json("a1", summary)
    # график: строки — задачи, колонки — критерии; отдельно для squares (Adam) и linear (OASIS)
    for update, fam in (("squares", "Adam"), ("linear", "OASIS")):
        fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 3.2 * len(tasks)))
        for i, (tname, _, _) in enumerate(tasks):
            for mode in ("l2", "wh", "w"):
                c = curves[f"{tname}|{update}|{mode}"]
                axes[i, 0].semilogy(c["t"], c["gF2"], color=COLORS[mode], label=fam + LBL[mode])
                axes[i, 1].semilogy(c["t"], c["gFt2"], color=COLORS[mode], label=fam + LBL[mode])
            axes[i, 0].set_ylabel(tname + "\n" + r"$\|\nabla F\|^2$")
            axes[i, 1].set_ylabel(r"$\|\nabla \tilde F_t\|^2$")
            for j in (0, 1):
                axes[i, j].grid(alpha=0.3)
            axes[i, 0].legend(fontsize=8)
        axes[-1, 0].set_xlabel("итерация")
        axes[-1, 1].set_xlabel("итерация")
        fig.suptitle(f"{fam}: критерий F vs критерий F̃")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"a1_{update}.png"), dpi=150)
        plt.close(fig)
    print("A1 done", flush=True)


# ================================================================ A2: плато vs beta (Т1/Т3)
def a2(device):
    """Дрейфовое плато delta ~ (1-beta) измеряется в СТОХАСТИЧЕСКОМ режиме: на полном
    градиенте v -> 0 у стационара, D замерзает на срезке и дрейф исчезает (это отдельно
    фиксируется как факт). Здесь: minibatch, delayed D; плато(beta) = C_noise + C_drift*(1-beta);
    вычитаем базлайн beta->1 и фитим наклон по (1-beta)."""
    reg = BoundedReg(0.1)
    betas = [0.9, 0.99, 0.999, 0.9999]
    beta_ref = 0.999999
    eta = 3e-3
    alpha, gamma = 1e-4, 10.0
    T = 400000
    seeds = (0, 1, 2)
    res = {"runs": [], "eta": eta, "alpha": alpha, "gamma": gamma}

    def plateau_runs(update, beta):
        plats, dDmaxs = [], []
        for s in seeds:
            pb = NonconvexClf(n=4096, d=60, seed=0, device=device, batch_size=64)
            _, D, lg = run_traj(pb, reg, "w", update, eta, beta, alpha, gamma,
                                T, seed=s, device=device, stochastic=True,
                                delayed=True, log_every=200)
            tail = np.array(lg["gFt2"][int(0.8 * len(lg["gFt2"])):])
            plats.append(float(np.mean(tail)))
            dDmaxs.append(float(np.max(lg["dD"][len(lg["dD"]) // 2:])))
        return plats, dDmaxs

    fits = {}
    for update in ("squares", "linear"):
        base_plats, _ = plateau_runs(update, beta_ref)
        base = float(np.median(base_plats))
        res["runs"].append({"update": update, "beta": beta_ref, "plateau_med": base,
                            "is_baseline": True})
        print("A2 baseline", update, base, flush=True)
        xs, ys = [], []
        for beta in betas:
            plats, dDmaxs = plateau_runs(update, beta)
            rho = (1 - beta) * gamma ** 2 / (2 * alpha) if update == "squares" else 2 * (1 - beta) * gamma
            rec = {"update": update, "beta": beta, "plateau_med": float(np.median(plats)),
                   "plateaus": plats, "excess_over_baseline": float(np.median(plats) - base),
                   "dD_max": float(np.max(dDmaxs)), "rho_beta_bound": rho}
            res["runs"].append(rec)
            print("A2", rec, flush=True)
            if rec["excess_over_baseline"] > 0:
                xs.append(np.log10(1 - beta))
                ys.append(np.log10(rec["excess_over_baseline"]))
        if len(xs) >= 3:
            k, b = np.polyfit(xs, ys, 1)
            fits[update] = {"slope_excess": float(k), "n_points": len(xs)}
    res["slope_fits"] = fits
    save_json("a2", res)
    # график: плато и превышение над шумовым базлайном
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, update in enumerate(("squares", "linear")):
        rr = [r for r in res["runs"] if r["update"] == update and not r.get("is_baseline")]
        base = [r for r in res["runs"] if r["update"] == update and r.get("is_baseline")][0]["plateau_med"]
        xs = [1 - r["beta"] for r in rr]
        axes[j].loglog(xs, [r["plateau_med"] for r in rr], "o-", label="плато")
        exc = [(1 - r["beta"], r["excess_over_baseline"]) for r in rr if r["excess_over_baseline"] > 0]
        if exc:
            axes[j].loglog([e[0] for e in exc], [e[1] for e in exc], "s--",
                           label=f"плато - базлайн (slope={fits.get(update, {}).get('slope_excess', float('nan')):.2f})")
        axes[j].axhline(base, color="gray", ls=":", label=r"базлайн $\beta\to1$ (шумовой пол)")
        axes[j].set_xlabel(r"$1-\beta$")
        axes[j].set_ylabel(r"плато $\mathbb{E}\|\nabla \tilde F\|^2$")
        axes[j].set_title({"squares": "Adam (eq:squares)", "linear": "OASIS (eq:linear)"}[update])
        axes[j].grid(alpha=0.3, which="both")
        axes[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a2_plateau.png"), dpi=150)
    plt.close(fig)
    # проверка леммы об эволюции D
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for update, mk in (("squares", "o"), ("linear", "s")):
        rr = [r for r in res["runs"] if r["update"] == update and not r.get("is_baseline")]
        ax.loglog([1 - r["beta"] for r in rr], [r["dD_max"] for r in rr], mk + "-",
                  label=f"max dD ({update})")
        ax.loglog([1 - r["beta"] for r in rr], [r["rho_beta_bound"] for r in rr], mk + "--",
                  alpha=0.5, label=f"граница леммы ({update})")
    ax.set_xlabel(r"$1-\beta$")
    ax.set_ylabel(r"$\max_t\|\hat D_{t+1}-\hat D_t\|_\infty$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a2_lemma_dD.png"), dpi=150)
    plt.close(fig)
    print("A2 done", fits, flush=True)


# ================================================================ A3: линейная скорость (Т2)
def a3(device):
    lam = 0.1
    reg = L2Reg(lam)
    configs = [
        {"kappa": 10, "alpha": 0.9, "gamma": 1.2},
        {"kappa": 10, "alpha": 0.5, "gamma": 2.0},
        {"kappa": 10, "alpha": 0.2, "gamma": 5.0},
    ]
    res = {"runs": []}
    fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 3.8))
    for ci, cfg in enumerate(configs):
        pb = DiagQuadratic(d=100, kappa=cfg["kappa"], device=device)
        alpha, gamma = cfg["alpha"], cfg["gamma"]
        mu_t = pb.mu_f + lam * alpha
        L_t = pb.L_f + lam * gamma
        eta_th = mu_t * alpha ** 2 / (gamma * L_t ** 2)
        # beta из условия теоремы: rho_beta = (1-beta)*Gamma^2/(2*alpha) <= eta*mu_t*alpha/(4*Gamma)
        beta = max(0.999, 1 - 0.9 * eta_th * mu_t * alpha ** 2 / (2 * gamma ** 3))
        rho_b = (1 - beta) * gamma ** 2 / (2 * alpha)
        ok_beta = rho_b <= eta_th * mu_t * alpha / (4 * gamma)
        for eta, tag in ((eta_th, "theory"), (min(50 * eta_th, 0.05), "practical")):
            T = int(min(3e6, 40 * 4 * gamma / (eta * mu_t)))
            w = w0_init(pb.d, 0, device)
            opt = PrecondOptimizer(pb, reg, mode="w", update="squares", eta=eta,
                                   beta2=beta, alpha=alpha, gamma=gamma, device=device)
            ts, R2s = [], []
            for t in range(T):
                w, info = opt.step(w)
                if t % max(1, T // 3000) == 0:
                    D = info["D_hat"]
                    wt = pb.wstar_tilde(lam, D)
                    R2 = float(((w - wt) * D * (w - wt)).sum())
                    ts.append(t)
                    R2s.append(R2)
            # эмпирическая скорость: наклон log(R2) на участке до плато
            R2a = np.array(R2s)
            mask = R2a > max(R2a.min() * 10, 1e-24)
            if mask.sum() > 10:
                k, _ = np.polyfit(np.array(ts)[mask], np.log(R2a[mask]), 1)
                rate_emp = -k
            else:
                rate_emp = float("nan")
            rate_th = eta * mu_t / (4 * gamma)
            rec = {"cfg": cfg, "eta": eta, "tag": tag, "T": T,
                   "rate_emp_per_iter": float(rate_emp), "rate_theory_per_iter": float(rate_th),
                   "ratio_emp_over_theory": float(rate_emp / rate_th) if rate_th > 0 else None,
                   "beta_condition_ok": bool(ok_beta), "R2_final": float(R2a[-1])}
            res["runs"].append(rec)
            print("A3", rec, flush=True)
            if tag == "theory":
                ax = axes[ci]
                # рисуем до выхода на плато + небольшой хвост
                stop = int(np.argmax(~mask)) if (~mask).any() else len(ts)
                stop = min(len(ts), int(stop * 1.15) + 5)
                tt = np.array(ts[:stop])
                ax.semilogy(tt, R2s[:stop], label="эксперимент")
                ax.semilogy(tt, R2s[0] * np.exp(-rate_th * tt), "--", alpha=0.6,
                            label="теор. скорость")
                ax.set_title(f"$\\alpha={alpha}$, $\\Gamma={gamma}$", fontsize=10)
                ax.set_xlabel("итерация")
                ax.grid(alpha=0.3)
                if ci == 0:
                    ax.set_ylabel(r"$\|w_t-\tilde w^*_t\|^2_{D_t}$")
                    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a3_linear_rate.png"), dpi=150)
    plt.close(fig)
    save_json("a3", res)
    print("A3 done", flush=True)


# ================================================================ A4: лемма lower bound
def a4(device):
    lams = np.logspace(-4, 0, 9)
    res = {"points": []}
    tasks = [
        ("diagquad10", DiagQuadratic(d=100, kappa=10, device=device)),
        ("diagquad1e3", DiagQuadratic(d=100, kappa=1e3, device=device)),
        ("mushrooms", LogReg(MUSHROOMS, device=device)),
    ]
    for tname, pb in tasks:
        for lam in lams:
            reg = L2Reg(float(lam))
            w, D, _ = run_traj(pb, reg, "w", "squares", eta=0.05, beta2=0.999,
                               alpha=1e-8, gamma=1e3, T=30000, freeze_at=20000,
                               device=device, log_every=1000)
            # точные решения
            if isinstance(pb, DiagQuadratic):
                wt = pb.wstar_tilde(float(lam), D)
                ws = pb.wstar_F(float(lam))
            else:
                wt, gn1 = newton_F(pb, reg, w0=w, D=D)
                ws, gn2 = newton_F(pb, reg)
            L_F = pb.L_f + float(lam)
            lhs = L_F * float((wt - ws).norm())
            rhs = float(((1 - D) * reg.grad(wt)).norm())
            res["points"].append({"task": tname, "lam": float(lam), "LHS": lhs, "RHS": rhs,
                                  "tight": rhs / lhs if lhs > 0 else None,
                                  "dist": float((wt - ws).norm()),
                                  "D_dist_I": float((D - 1).abs().max())})
            print("A4", res["points"][-1], flush=True)
    viol = [p for p in res["points"] if p["RHS"] > p["LHS"] * (1 + 1e-8)]
    res["violations"] = viol
    save_json("a4", res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = {"diagquad10": r"квадратичная, $\kappa=10$",
             "diagquad1e3": r"квадратичная, $\kappa=10^3$",
             "mushrooms": "mushrooms"}
    for tname, mk in (("diagquad10", "o"), ("diagquad1e3", "s"), ("mushrooms", "^")):
        pts = [p for p in res["points"] if p["task"] == tname]
        axes[0].loglog([p["RHS"] for p in pts], [p["LHS"] for p in pts], mk, label=names[tname])
        axes[1].semilogx([p["lam"] for p in pts], [p["tight"] for p in pts], mk + "-", label=names[tname])
    lims = axes[0].get_xlim()
    axes[0].loglog(lims, lims, "k--", alpha=0.5, label="$y = x$")
    axes[0].set_xlabel(r"$\|(I-D_t)\,\nabla r(\widetilde w^*_t)\|$")
    axes[0].set_ylabel(r"$L_F\,\|\widetilde w^*_t-w^*\|$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("отношение правой части к левой")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a4_lemma.png"), dpi=150)
    plt.close(fig)
    print("A4 done, violations:", len(viol), flush=True)


# ================================================================ A5: распределение D_inf
def a5(device):
    pb = LogReg(MUSHROOMS, device=device)
    lam = 0.04
    reg = L2Reg(lam)
    out = {}
    finals = {}
    for mode in ("w", "l2"):
        w, D, _ = run_traj(pb, reg, mode, "squares", eta=0.05, beta2=0.999,
                           alpha=1e-8, gamma=1e3, T=30000, freeze_at=20000,
                           device=device, log_every=1000)
        finals[mode] = (w.cpu().numpy(), D.cpu().numpy())
    freq = (pb.X != 0).to(torch.float64).mean(0).cpu().numpy()
    d_w = finals["w"][1]
    from scipy.stats import spearmanr
    corr, pval = spearmanr(freq, d_w)
    out["spearman_d_vs_freq"] = {"corr": float(corr), "p": float(pval)}
    # сравнение |w| на редких признаках
    rare = freq < np.median(freq)
    out["mean_abs_w_rare"] = {
        "adamw": float(np.abs(finals["w"][0][rare]).mean()),
        "adaml2": float(np.abs(finals["l2"][0][rare]).mean()),
    }
    out["mean_abs_w_freq"] = {
        "adamw": float(np.abs(finals["w"][0][~rare]).mean()),
        "adaml2": float(np.abs(finals["l2"][0][~rare]).mean()),
    }
    save_json("a5", out)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(np.log10(d_w), bins=40)
    axes[0].set_xlabel(r"$\log_{10} d^i$ (AdamW, стационарный участок)")
    axes[0].set_ylabel("число координат")
    axes[1].loglog(freq + 1e-6, d_w, ".", alpha=0.6)
    axes[1].set_xlabel("частота признака в выборке")
    axes[1].set_ylabel(r"$d^i$")
    axes[2].loglog(np.abs(finals["l2"][0]) + 1e-12, np.abs(finals["w"][0]) + 1e-12, ".", alpha=0.6)
    lims = axes[2].get_xlim()
    axes[2].loglog(lims, lims, "k--", alpha=0.5, label="$y = x$")
    axes[2].set_xlabel(r"$|w^{*,i}|$, AdamL2")
    axes[2].set_ylabel(r"$|\widetilde w^{*,i}|$, AdamW")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a5_dinf.png"), dpi=150)
    plt.close(fig)
    print("A5 done", out, flush=True)


# ================================================================ A6: куда сходится WH
def a6(device):
    """(1) стационарная точка WH — решение исходной задачи F (как у L2, но с чистым D);
    (2) неустойчивость WH на мёртвых координатах: шаг по регуляризатору ~ eta*lam/alpha,
    расходимость при eta*lam/alpha > 2 — проверяем порог количественно."""
    res = {}
    ALPHA = 1e-2  # для устойчивости WH: eta*lam/ALPHA < 2
    tasks = [
        ("diagquad", DiagQuadratic(d=100, kappa=1e3, device=device), [3e-4, 1e-3, 3e-3], 200000),
        ("mushrooms", LogReg(MUSHROOMS, device=device), [3e-3, 1e-2, 3e-2], 40000),
    ]
    for tname, pb, etas, T in tasks:
        for lam in (1e-2, 1e-1):
            reg = L2Reg(lam)
            ws_exact, _ = newton_F(pb, reg)
            row = {}
            for mode in ("l2", "wh", "w"):
                crit_idx = 1 if mode == "w" else 0  # w: по gFt2, иначе по gF2
                best = None
                for eta in etas:
                    w, D, lg = run_traj(pb, reg, mode, "squares", eta=eta, beta2=0.999,
                                        alpha=ALPHA, gamma=1e3, T=T, device=device,
                                        log_every=max(1, T // 1000))
                    c = criteria(pb, reg, w, D)
                    if not np.isfinite(c[crit_idx]):
                        continue
                    if best is None or c[crit_idx] < best[0]:
                        best = (c[crit_idx], eta, c, float((w - ws_exact).norm()))
                if best is None:
                    row[mode] = {"gF2_final": float("nan"), "gFt2_final": float("nan"),
                                 "dist_to_wstarF": float("nan"), "eta": None}
                else:
                    row[mode] = {"gF2_final": best[2][0], "gFt2_final": best[2][1],
                                 "dist_to_wstarF": best[3], "eta": best[1]}
            res[f"{tname}|lam={lam}"] = row
            print("A6", tname, lam, {m: (round(np.log10(max(v['gF2_final'], 1e-300)), 1)
                                          if np.isfinite(v['gF2_final']) else 'nan') for m, v in row.items()}, flush=True)
    # порог неустойчивости WH: расходимость <=> eta*lam/alpha > 2
    pb = LogReg(MUSHROOMS, device=device)
    lam, eta = 0.1, 0.03
    inst = []
    for alpha in (1e-8, 1e-4, 3e-3, 1e-2, 1e-1):
        reg = L2Reg(lam)
        w, D, _ = run_traj(pb, reg, "wh", "squares", eta=eta, beta2=0.999,
                           alpha=alpha, gamma=1e3, T=5000, device=device, log_every=500)
        diverged = bool(torch.isnan(w).any() or w.norm() > 1e6)
        inst.append({"alpha": alpha, "eta_lam_over_alpha": eta * lam / alpha,
                     "predicted_unstable": eta * lam / alpha > 2, "diverged": diverged})
        print("A6-instability", inst[-1], flush=True)
    res["wh_instability_threshold"] = inst
    save_json("a6", res)
    # график сходимости по ||grad F||^2 для трех режимов (mushrooms, lam=0.1)
    pb = LogReg(MUSHROOMS, device=device)
    reg = L2Reg(0.1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode in ("l2", "wh", "w"):
        eta_m = res["mushrooms|lam=0.1"][mode].get("eta") or 0.01
        _, _, lg = run_traj(pb, reg, mode, "squares", eta=eta_m, beta2=0.999,
                            alpha=ALPHA, gamma=1e3, T=40000, device=device, log_every=40)
        y0 = np.array(lg["gF2"]); y0[~np.isfinite(y0)] = np.nan
        y1 = np.array(lg["gFt2"]); y1[~np.isfinite(y1)] = np.nan
        axes[0].semilogy(lg["t"], y0, color=COLORS[mode], label="Adam" + LBL[mode])
        axes[1].semilogy(lg["t"], y1, color=COLORS[mode], label="Adam" + LBL[mode])
    axes[0].set_ylabel(r"$\|\nabla F\|^2$")
    axes[1].set_ylabel(r"$\|\nabla \tilde F_t\|^2$")
    for a in axes:
        a.set_xlabel("итерация")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Куда сходится WH: у WH и L2 убывает grad F, у W — только grad F̃")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a6_wh.png"), dpi=150)
    plt.close(fig)
    print("A6 done", flush=True)


# ================================================================ S1: стохастика, Т3
def s1(device):
    """Плато E||grad F_tilde||^2 ~ Gamma*L*eta*sigma^2/alpha^2: наклон 1 по eta и по 1/batch."""
    pb_full = NonconvexClf(n=4096, d=60, seed=0, device=device)
    reg = BoundedReg(0.1)
    batches = [8, 64, 512, 4096]
    etas = [1e-3, 3e-3, 1e-2]
    beta = 0.9999
    T = 200000
    res = {"runs": []}
    for bs in batches:
        pb = NonconvexClf(n=4096, d=60, seed=0, device=device, batch_size=None if bs >= 4096 else bs)
        for eta in etas:
            plats = []
            for s in (0, 1, 2):
                _, D, lg = run_traj(pb, reg, "w", "squares", eta, beta, 1e-4, 10.0,
                                    T, seed=s, device=device, stochastic=True,
                                    delayed=True, log_every=100)
                tail = np.array(lg["gFt2"][int(0.8 * len(lg["gFt2"])):])
                plats.append(float(np.mean(tail)))
            # эмпирическая дисперсия стох. градиента в конечной точке
            rng = torch.Generator(device=device).manual_seed(123)
            wf = w0_init(pb.d, 0, device)  # не важно где — оценим в 0-й точке трэка? используем послед. w
            res["runs"].append({"batch": bs, "eta": eta, "plateau": float(np.median(plats)),
                                "plateaus": plats})
            print("S1", res["runs"][-1], flush=True)
    # фиты
    fits = {}
    for bs in batches:
        xs = np.log10([r["eta"] for r in res["runs"] if r["batch"] == bs])
        ys = np.log10([r["plateau"] for r in res["runs"] if r["batch"] == bs])
        k, _ = np.polyfit(xs, ys, 1)
        fits[f"slope_vs_eta|batch={bs}"] = float(k)
    for eta in etas:
        xs = np.log10([1.0 / r["batch"] for r in res["runs"] if r["eta"] == eta and r["batch"] < 4096])
        ys = np.log10([r["plateau"] for r in res["runs"] if r["eta"] == eta and r["batch"] < 4096])
        k, _ = np.polyfit(xs, ys, 1)
        fits[f"slope_vs_invbatch|eta={eta}"] = float(k)
    res["fits"] = fits
    save_json("s1", res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for bs, mk in zip(batches, ("o", "s", "^", "d")):
        xs = [r["eta"] for r in res["runs"] if r["batch"] == bs]
        ys = [r["plateau"] for r in res["runs"] if r["batch"] == bs]
        axes[0].loglog(xs, ys, mk + "-", label=f"batch={bs}")
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"плато $\mathbb{E}\|\nabla\tilde F\|^2$")
    axes[0].set_title("Т3: плато ~ eta*sigma^2 (наклон 1 по eta)")
    for eta, mk in zip(etas, ("o", "s", "^")):
        xs = [1.0 / r["batch"] for r in res["runs"] if r["eta"] == eta]
        ys = [r["plateau"] for r in res["runs"] if r["eta"] == eta]
        axes[1].loglog(xs, ys, mk + "-", label=f"eta={eta}")
    axes[1].set_xlabel("1/batch  (~sigma^2)")
    axes[1].set_title("Т3: плато ~ 1/batch")
    for a in axes:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "s1_stoch_plateau.png"), dpi=150)
    plt.close(fig)
    print("S1 done", fits, flush=True)


# ================================================================ A7: мосты теория <-> практика
def a7(device):
    """(a) срезка alpha vs практический eps (sqrt(v)+eps); (b) абляция beta1 и bias-correction;
    (c) стабилизация D_t (solution.tex ссылается на эту проверку);
    (d) наклоны констант леммы по Gamma и alpha (стохастический режим)."""
    res = {}
    pb = LogReg(MUSHROOMS, device=device)
    lam = 0.04
    reg = L2Reg(lam)

    # ---- (a) clamp(alpha) vs eps-режим
    part_a = {}
    for tag, kw in (("clamp_1e-8", dict(alpha=1e-8, eps_mode=False)),
                    ("eps_1e-8", dict(alpha=1e-8, eps_mode=True)),
                    ("clamp_1e-4", dict(alpha=1e-4, eps_mode=False)),
                    ("eps_1e-4", dict(alpha=1e-4, eps_mode=True))):
        w = w0_init(pb.d, 0, device)
        opt = PrecondOptimizer(pb, reg, mode="w", update="squares", eta=0.03,
                               beta2=0.999, gamma=1e3, device=device, **kw)
        D = None
        traj = []
        for t in range(20000):
            w, info = opt.step(w)
            D = info["D_hat"]
            if t % 20 == 0:
                traj.append(criteria(pb, reg, w, D))
        part_a[tag] = {"gF2_final": traj[-1][0], "gFt2_final": traj[-1][1],
                       "gFt2_drop_orders": float(np.log10(max(traj[0][1], 1e-300) / max(traj[-1][1], 1e-300)))}
        print("A7a", tag, part_a[tag], flush=True)
    res["a_alpha_vs_eps"] = part_a

    # ---- (b) beta1 и bias-correction
    part_b = {}
    for b1 in (0.0, 0.9):
        for bc in (False, True):
            w = w0_init(pb.d, 0, device)
            opt = PrecondOptimizer(pb, reg, mode="w", update="squares", eta=0.03,
                                   beta1=b1, beta2=0.999, alpha=1e-8, gamma=1e3,
                                   bias_correction=bc, device=device)
            D = None
            for t in range(20000):
                w, info = opt.step(w)
                D = info["D_hat"]
            gF2, gFt2 = criteria(pb, reg, w, D)
            part_b[f"beta1={b1}|bc={bc}"] = {"gF2_final": gF2, "gFt2_final": gFt2}
            print("A7b", f"beta1={b1} bc={bc}", part_b[f"beta1={b1}|bc={bc}"], flush=True)
    res["b_beta1_bias_correction"] = part_b

    # ---- (c) стабилизация D_t: ||D_t - D_T||inf / ||D_T||inf
    part_c = {}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag, bs in (("full-batch", None), ("batch=128", 128)):
        pbc = LogReg(MUSHROOMS, device=device, batch_size=bs)
        w = w0_init(pbc.d, 0, device)
        opt = PrecondOptimizer(pbc, reg, mode="w", update="squares", eta=0.03,
                               beta2=0.999, alpha=1e-8, gamma=1e3,
                               delayed=(bs is not None), device=device)
        Ds, ts = [], []
        T = 30000
        for t in range(T):
            w, info = opt.step(w, stochastic=(bs is not None))
            if t % 100 == 0:
                Ds.append(info["D_hat"].clone())
                ts.append(t)
        DT = Ds[-1]
        relerr = [float((Dt - DT).abs().max() / DT.abs().max()) for Dt in Ds]
        part_c[tag] = {"rel_dist_half": relerr[len(relerr) // 2], "rel_dist_090": relerr[int(0.9 * len(relerr))]}
        ax.semilogy(ts[:-1], np.maximum(relerr[:-1], 1e-17), label=tag)
        print("A7c", tag, part_c[tag], flush=True)
    ax.set_xlabel("итерация")
    ax.set_ylabel(r"$\|\hat D_t-\hat D_T\|_\infty / \|\hat D_T\|_\infty$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "a7_dstab.png"), dpi=150)
    plt.close(fig)
    res["c_D_stabilization"] = part_c

    # ---- (d) наклоны dD_max по Gamma и alpha (стохастический режим, лемма)
    part_d = {"runs": []}
    def dd_max(update, alpha, gamma):
        pbn = NonconvexClf(n=4096, d=60, seed=0, device=device, batch_size=64)
        _, _, lg = run_traj(pbn, BoundedReg(0.1), "w", update, 3e-3, 0.99, alpha, gamma,
                            100000, seed=0, device=device, stochastic=True,
                            delayed=True, log_every=100)
        return float(np.max(lg["dD"][len(lg["dD"]) // 2:]))
    for update in ("squares", "linear"):
        for gamma in (2.0, 5.0, 10.0, 20.0):
            v = dd_max(update, 1e-3, gamma)
            part_d["runs"].append({"update": update, "vary": "gamma", "gamma": gamma,
                                   "alpha": 1e-3, "dD_max": v})
            print("A7d", part_d["runs"][-1], flush=True)
        for alpha in (1e-4, 1e-3, 1e-2):
            v = dd_max(update, alpha, 10.0)
            part_d["runs"].append({"update": update, "vary": "alpha", "gamma": 10.0,
                                   "alpha": alpha, "dD_max": v})
            print("A7d", part_d["runs"][-1], flush=True)
    for update in ("squares", "linear"):
        g = [(np.log10(r["gamma"]), np.log10(r["dD_max"])) for r in part_d["runs"]
             if r["update"] == update and r["vary"] == "gamma" and r["dD_max"] > 0]
        a = [(np.log10(r["alpha"]), np.log10(r["dD_max"])) for r in part_d["runs"]
             if r["update"] == update and r["vary"] == "alpha" and r["dD_max"] > 0]
        if len(g) >= 3:
            part_d[f"slope_vs_gamma|{update}"] = float(np.polyfit(*zip(*g), 1)[0])
        if len(a) >= 3:
            part_d[f"slope_vs_alpha|{update}"] = float(np.polyfit(*zip(*a), 1)[0])
    res["d_lemma_constants"] = part_d
    save_json("a7", res)
    print("A7 done", flush=True)


# ================================================================ S1b: стохастика, Т3 (чистый eta-скейлинг)
def s1b(device):
    """Чистая проверка шумового члена Т3 (Gamma*L*eta*sigma^2/alpha^2): OASIS-тип на
    квадратичной задаче с аддитивным шумом — D = diag(гессиан) constant, самонормализация
    Adam-типа отсутствует, плато E||grad F_tilde||^2 должно расти как eta*sigma^2."""
    reg = BoundedReg(0.1)
    etas = [3e-4, 1e-3, 3e-3, 1e-2]
    sigmas = [0.1, 0.3, 1.0]
    beta = 0.9999
    T = 200000
    res = {"runs": []}
    for sigma in sigmas:
        pb = DiagQuadratic(d=100, kappa=10, device=device, noise_sigma=sigma)
        for eta in etas:
            plats = []
            for s in (0, 1, 2):
                _, D, lg = run_traj(pb, reg, "w", "linear", eta, beta, 0.5, 20.0,
                                    T, seed=s, device=device, stochastic=True,
                                    delayed=True, log_every=100)
                tail = np.array(lg["gFt2"][int(0.8 * len(lg["gFt2"])):])
                plats.append(float(np.mean(tail)))
            res["runs"].append({"sigma": sigma, "eta": eta, "plateau": float(np.median(plats))})
            print("S1b", res["runs"][-1], flush=True)
    fits = {}
    for sigma in sigmas:
        xs = np.log10([r["eta"] for r in res["runs"] if r["sigma"] == sigma])
        ys = np.log10([r["plateau"] for r in res["runs"] if r["sigma"] == sigma])
        k, _ = np.polyfit(xs, ys, 1)
        fits[f"slope_vs_eta|sigma={sigma}"] = float(k)
    for eta in etas:
        xs = np.log10([r["sigma"] ** 2 for r in res["runs"] if r["eta"] == eta])
        ys = np.log10([r["plateau"] for r in res["runs"] if r["eta"] == eta])
        k, _ = np.polyfit(xs, ys, 1)
        fits[f"slope_vs_sigma2|eta={eta}"] = float(k)
    res["fits"] = fits
    save_json("s1b", res)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sigma, mk in zip(sigmas, ("o", "s", "^")):
        xs = [r["eta"] for r in res["runs"] if r["sigma"] == sigma]
        ys = [r["plateau"] for r in res["runs"] if r["sigma"] == sigma]
        ax.loglog(xs, ys, mk + "-",
                  label=f"sigma={sigma}, slope={fits[f'slope_vs_eta|sigma={sigma}']:.2f}")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"плато $\mathbb{E}\|\nabla\tilde F\|^2$")
    ax.set_title("Т3, шумовой член: плато ~ eta*sigma^2 (OASIS-тип, D = const)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "s1b_eta_scaling.png"), dpi=150)
    plt.close(fig)
    print("S1b done", fits, flush=True)


# ================================================================ S2: стохастика, Т4
def s2(device):
    """Шумовой шар: steady-state E||w - w~*||^2_D ~ 8*Gamma*eta*sigma^2/(mu~*alpha)."""
    lam = 0.1
    reg = L2Reg(lam)
    alpha, gamma = 0.5, 2.0
    beta = 0.9999
    etas = [1e-3, 3e-3, 1e-2, 3e-2]
    sigmas = [0.1, 0.3, 1.0]
    T = 200000
    res = {"runs": []}
    for sigma in sigmas:
        pb = DiagQuadratic(d=100, kappa=10, device=device, noise_sigma=sigma)
        mu_t = pb.mu_f + lam * alpha
        for eta in etas:
            balls = []
            for s in (0, 1, 2):
                w = w0_init(pb.d, s, device)
                opt = PrecondOptimizer(pb, reg, mode="w", update="squares", eta=eta,
                                       beta2=beta, alpha=alpha, gamma=gamma,
                                       delayed=True, seed=s, device=device)
                R2s = []
                for t in range(T):
                    w, info = opt.step(w, stochastic=True)
                    if t % 100 == 0 and t > 0.8 * T:
                        D = info["D_hat"]
                        wt = pb.wstar_tilde(lam, D)
                        R2s.append(float(((w - wt) * D * (w - wt)).sum()))
                balls.append(float(np.mean(R2s)))
            pred = 8 * gamma * eta * sigma ** 2 / (mu_t * alpha)
            rec = {"sigma": sigma, "eta": eta, "ball_emp": float(np.median(balls)),
                   "ball_theory_bound": pred, "within_bound": bool(np.median(balls) <= pred)}
            res["runs"].append(rec)
            print("S2", rec, flush=True)
    fits = {}
    for sigma in sigmas:
        xs = np.log10([r["eta"] for r in res["runs"] if r["sigma"] == sigma])
        ys = np.log10([r["ball_emp"] for r in res["runs"] if r["sigma"] == sigma])
        k, _ = np.polyfit(xs, ys, 1)
        fits[f"slope_vs_eta|sigma={sigma}"] = float(k)
    res["fits"] = fits
    save_json("s2", res)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sigma, mk in zip(sigmas, ("o", "s", "^")):
        xs = [r["eta"] for r in res["runs"] if r["sigma"] == sigma]
        ys = [r["ball_emp"] for r in res["runs"] if r["sigma"] == sigma]
        bs = [r["ball_theory_bound"] for r in res["runs"] if r["sigma"] == sigma]
        ax.loglog(xs, ys, mk + "-", label=f"sigma={sigma} (emp), slope={fits[f'slope_vs_eta|sigma={sigma}']:.2f}")
        ax.loglog(xs, bs, mk + "--", alpha=0.4, color=ax.lines[-1].get_color())
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"steady-state $\mathbb{E}\|w-\tilde w^*\|^2_{D}$")
    ax.set_title("Т4: шумовой шар ~ eta*sigma^2 (пунктир — верхняя граница теоремы)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "s2_noise_ball.png"), dpi=150)
    plt.close(fig)
    print("S2 done", fits, flush=True)


# ================================================================ CLI
EXPS = {"a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6,
        "a7": a7, "s1": s1, "s1b": s1b, "s2": s2}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(EXPS))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_default_dtype(torch.float64)
    EXPS[args.exp](args.device)
    print(f"[{args.exp}] elapsed {time.time() - t0:.1f}s", flush=True)
