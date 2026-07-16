"""Единый оптимизатор с предобуславливанием и тремя режимами регуляризации.

Режимы (цвета из Алгоритма 1 диплома):
  'l2' (синий):  g <- grad f + grad r; D строится по g;      w -= eta * D^-1 m
  'wh' (оранж.): g <- grad f;          D по g;               w -= eta * D^-1 (m + grad r)
  'w'  (красн.): g <- grad f;          D по g;               w -= eta * D^-1 m + eta * grad r

Обновление D:
  'squares' (Adam):  v = beta2*v + (1-beta2)*h^2,  D_raw = sqrt(v)   (eq:squares, H = diag|h|)
  'linear'  (OASIS): D_raw = beta2*D_raw + (1-beta2)*(z * H z)       (eq:linear, Хатчинсон)

Срезка (eq:alpha + верхняя граница для контролируемости Gamma):
  D_hat = clamp(|D_raw|, alpha, Gamma)

delayed=True: D_hat для шага t строится по информации до g_t (предсказуемость из ass:expectations).
"""
import torch


class PrecondOptimizer:
    def __init__(self, problem, reg, mode="w", update="squares", eta=1e-2,
                 beta1=0.0, beta2=0.999, alpha=1e-8, gamma=1e3,
                 bias_correction=False, delayed=False, exact_diag_hess=True,
                 eps_mode=False, seed=0, device="cpu"):
        assert mode in ("l2", "w", "wh") and update in ("squares", "linear")
        self.pb, self.reg, self.mode, self.update = problem, reg, mode, update
        self.eta, self.beta1, self.beta2 = eta, beta1, beta2
        self.alpha, self.gamma = alpha, gamma
        self.bias_correction, self.delayed = bias_correction, delayed
        self.exact_diag_hess = exact_diag_hess
        self.eps_mode = eps_mode  # практический Adam: D = sqrt(v) + alpha вместо clamp
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        d = problem.d
        self.m = torch.zeros(d, dtype=torch.float64, device=device)
        self.v = torch.zeros(d, dtype=torch.float64, device=device)      # для squares
        self.D_raw = torch.ones(d, dtype=torch.float64, device=device)   # для linear
        self.t = 0
        self.D_hat_prev = None

    # -------------------------------------------------- внутренние блоки
    def _hutchinson(self, w):
        """diag-оценка гессиана: z * (H z), z из Радемахера; для скорости на наших
        задачах доступен и точный диагональный гессиан (exact_diag_hess=True)."""
        if self.exact_diag_hess:
            h = self.pb.diag_hess_f(w)
            if self.mode == "l2":
                h = h + self.reg.diag_hess(w)
            return h
        z = torch.randint(0, 2, (self.pb.d,), generator=self.rng, device=w.device,
                          dtype=torch.float64) * 2 - 1
        H = self.pb.hess_f(w)
        if self.mode == "l2":
            Hz = H @ z + self.reg.diag_hess(w) * z
        else:
            Hz = H @ z
        return z * Hz

    def _D_hat_from_state(self):
        if self.update == "squares":
            v = self.v / (1 - self.beta2 ** max(self.t, 1)) if self.bias_correction else self.v
            D_raw = torch.sqrt(v)
            if self.eps_mode:
                return D_raw + self.alpha  # sqrt(v) + eps, как в практическом Adam
        else:
            D_raw = self.D_raw
        return torch.clamp(D_raw.abs(), min=self.alpha, max=self.gamma)

    def _update_D_state(self, w, g_for_D):
        if self.update == "squares":
            self.v = self.beta2 * self.v + (1 - self.beta2) * g_for_D * g_for_D
        else:
            h = self._hutchinson(w)
            self.D_raw = self.beta2 * self.D_raw + (1 - self.beta2) * h

    # -------------------------------------------------- шаг
    def step(self, w, stochastic=False):
        """Возвращает (w_new, info). info содержит D_hat, использованную в шаге."""
        self.t += 1
        g_f = self.pb.stoch_grad(w, self.rng) if stochastic else self.pb.grad_f(w)
        g_r = self.reg.grad(w)
        g_for_D = g_f + g_r if self.mode == "l2" else g_f

        if self.delayed:
            D_hat = self._D_hat_from_state()          # по информации до g_t
            self._update_D_state(w, g_for_D)
        else:
            self._update_D_state(w, g_for_D)
            D_hat = self._D_hat_from_state()

        if self.beta1 > 0:
            self.m = self.beta1 * self.m + (1 - self.beta1) * g_for_D
            m = self.m / (1 - self.beta1 ** self.t) if self.bias_correction else self.m
        else:
            m = g_for_D

        if self.mode == "l2":
            w_new = w - self.eta * m / D_hat
        elif self.mode == "wh":
            w_new = w - self.eta * (m + g_r) / D_hat
        else:  # 'w' — затухание весов
            w_new = w - self.eta * m / D_hat - self.eta * g_r

        dD = None
        if self.D_hat_prev is not None:
            dD = float((D_hat - self.D_hat_prev).abs().max())
        self.D_hat_prev = D_hat.clone()
        return w_new, {"D_hat": D_hat, "dD_inf": dD}


# -------------------------------------------------- критерии
def criteria(problem, reg, w, D_hat):
    """||grad F||^2 и ||grad F_tilde||^2 с той же D_hat, что в шаге."""
    gf = problem.grad_f(w)
    gr = reg.grad(w)
    gF = gf + gr
    gFt = gf + D_hat * gr
    return float((gF * gF).sum()), float((gFt * gFt).sum())
