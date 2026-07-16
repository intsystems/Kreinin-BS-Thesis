"""Точные решения w* (min F) и w~* (min f + r~ при замороженной D) методом Ньютона."""
import torch


def newton_F(problem, reg, w0=None, D=None, tol=1e-12, max_iter=200):
    """Решает grad f(w) + M grad r(w) = 0, где M = diag(D) (или I, если D=None).

    Демпфированный Ньютон с гессианом grad^2 f + M * diag_hess(r).
    Возвращает (w, ||grad||).
    """
    d = problem.d
    device = getattr(problem, "device", "cpu")
    w = torch.zeros(d, dtype=torch.float64, device=device) if w0 is None else w0.clone()
    Mv = torch.ones(d, dtype=torch.float64, device=device) if D is None else D

    def G(w):
        return problem.grad_f(w) + Mv * reg.grad(w)

    for _ in range(max_iter):
        g = G(w)
        gn = float(g.norm())
        if gn < tol:
            break
        H = problem.hess_f(w) + torch.diag(Mv * reg.diag_hess(w))
        # регуляризация гессиана на случай вырожденности/невыпуклости r
        H = H + 1e-14 * torch.eye(d, dtype=torch.float64, device=device)
        try:
            step = torch.linalg.solve(H, g)
        except Exception:
            step = g / (torch.diagonal(H).abs() + 1e-12)
        # затухающий шаг: уменьшаем, пока норма градиента не падает
        lam = 1.0
        for _ls in range(60):
            w_new = w - lam * step
            if float(G(w_new).norm()) < gn:
                break
            lam *= 0.5
        w = w_new
    return w, float(G(w).norm())


def constants(problem, reg):
    """L_f, mu_f, L_F = L_f + L_r."""
    return {
        "L_f": problem.L_f,
        "mu_f": problem.mu_f,
        "L_r": float(reg.L_r),
        "L_F": problem.L_f + float(reg.L_r),
    }
