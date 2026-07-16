"""Задачи для количественной проверки теории (float64, полный/стохастический градиент).

Все задачи предоставляют:
  f(w), grad_f(w), hess_f(w) (полный гессиан, d x d), diag_hess_f(w),
  stoch_grad(w, rng)  — стохастический градиент (минибатч или аддитивный шум),
  константы L_f, mu_f.

Регуляризаторы:
  l2:      r(w) = lam/2 ||w||^2          (Тихонов; для Т2/Т4)
  bounded: r(w) = lam * sum x^2/(1+x^2)  (ограниченный сепарабельный; для Т1/Т3, |r| <= lam*d)
"""
import numpy as np
import torch


# ---------------------------------------------------------------- регуляризаторы
class L2Reg:
    name = "l2"

    def __init__(self, lam):
        self.lam = lam

    def value(self, w):
        return 0.5 * self.lam * (w * w).sum()

    def grad(self, w):
        return self.lam * w

    def diag_hess(self, w):
        return torch.full_like(w, self.lam)

    @property
    def L_r(self):
        return self.lam

    def omega(self, d):
        return float("inf")


class BoundedReg:
    """r(w) = lam * sum_i w_i^2 / (1 + w_i^2): сепарабельный, гладкий, 0 <= r <= lam*d."""
    name = "bounded"

    def __init__(self, lam):
        self.lam = lam

    def value(self, w):
        return self.lam * (w * w / (1 + w * w)).sum()

    def grad(self, w):
        return self.lam * 2 * w / (1 + w * w) ** 2

    def diag_hess(self, w):
        w2 = w * w
        return self.lam * 2 * (1 - 3 * w2) / (1 + w2) ** 3

    @property
    def L_r(self):
        return 2 * self.lam  # sup |d^2/dx^2 [x^2/(1+x^2)]| = 2

    def omega(self, d):
        return self.lam * d


# ---------------------------------------------------------------- задачи
class DiagQuadratic:
    """f(w) = 1/2 sum a_i w_i^2 - b_i w_i. Все решения аналитические."""

    def __init__(self, d=100, kappa=1e3, seed=0, device="cpu", noise_sigma=0.0):
        g = torch.Generator().manual_seed(seed)
        loga = torch.linspace(0, np.log10(kappa), d)
        self.a = (10 ** loga).to(torch.float64).to(device)  # спектр [1, kappa]
        self.b = torch.randn(d, generator=g, dtype=torch.float64).to(device)
        self.d, self.device = d, device
        self.noise_sigma = noise_sigma
        self.L_f = float(self.a.max())
        self.mu_f = float(self.a.min())
        self.name = f"diagquad_k{kappa:g}"

    def f(self, w):
        return (0.5 * self.a * w * w - self.b * w).sum()

    def grad_f(self, w):
        return self.a * w - self.b

    def diag_hess_f(self, w):
        return self.a.clone()

    def hess_f(self, w):
        return torch.diag(self.a)

    def stoch_grad(self, w, rng):
        g = self.grad_f(w)
        if self.noise_sigma > 0:
            xi = torch.randn(self.d, generator=rng, dtype=torch.float64, device=w.device)
            g = g + self.noise_sigma * xi / np.sqrt(self.d)  # E||xi_scaled||^2 = sigma^2
        return g

    def wstar_F(self, lam):
        return self.b / (self.a + lam)

    def wstar_tilde(self, lam, D):
        return self.b / (self.a + lam * D)


class Quadratic:
    """f(w) = 1/2 w^T A w - b^T w, A = Q diag(spec) Q^T, лог-равномерный спектр [1, kappa]."""

    def __init__(self, d=100, kappa=1e3, seed=0, device="cpu", noise_sigma=0.0):
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(d, d, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(M)
        spec = 10 ** torch.linspace(0, np.log10(kappa), d).to(torch.float64)
        self.A = (Q * spec.unsqueeze(0)) @ Q.T
        self.A = (0.5 * (self.A + self.A.T)).to(device)
        self.b = torch.randn(d, generator=g, dtype=torch.float64).to(device)
        self.d, self.device = d, device
        self.noise_sigma = noise_sigma
        self.L_f = float(spec.max())
        self.mu_f = float(spec.min())
        self.name = f"quad_k{kappa:g}"

    def f(self, w):
        return 0.5 * w @ (self.A @ w) - self.b @ w

    def grad_f(self, w):
        return self.A @ w - self.b

    def diag_hess_f(self, w):
        return torch.diagonal(self.A).clone()

    def hess_f(self, w):
        return self.A

    def stoch_grad(self, w, rng):
        g = self.grad_f(w)
        if self.noise_sigma > 0:
            xi = torch.randn(self.d, generator=rng, dtype=torch.float64, device=w.device)
            g = g + self.noise_sigma * xi / np.sqrt(self.d)
        return g


class LogReg:
    """Логистическая регрессия: f(w) = 1/n sum log(1 + exp(-y_i x_i^T w))."""

    def __init__(self, path, device="cpu", batch_size=None):
        from sklearn.datasets import load_svmlight_file
        X, y = load_svmlight_file(path)
        X = torch.tensor(X.toarray(), dtype=torch.float64, device=device)
        y = torch.tensor(y, dtype=torch.float64, device=device)
        uniq = torch.unique(y)
        y = torch.where(y == uniq[0], -1.0, 1.0).to(torch.float64)
        self.X, self.y = X, y
        self.n, self.d = X.shape
        self.device = device
        self.batch_size = batch_size
        # L_f = lambda_max(X^T X) / (4n)
        sv = torch.linalg.matrix_norm(X, ord=2)
        self.L_f = float(sv ** 2 / (4 * self.n))
        self.mu_f = 0.0
        self.name = path.split("/")[-1].split(".")[0]

    def _margins(self, w, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        return y * (X @ w)

    def f(self, w):
        m = self._margins(w)
        return torch.nn.functional.softplus(-m).mean()

    def grad_f(self, w, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        m = y * (X @ w)
        s = torch.sigmoid(-m)  # = 1 - sigma(m)
        return -(X.T @ (y * s)) / X.shape[0]

    def diag_hess_f(self, w, X=None):
        X = self.X if X is None else X
        m = self._margins(w) if X is self.X else None
        if m is None:
            m = self.y[: X.shape[0]] * (X @ w)
        p = torch.sigmoid(m) * torch.sigmoid(-m)
        return (X * X * p.unsqueeze(1)).sum(0) / X.shape[0]

    def hess_f(self, w):
        m = self._margins(w)
        p = torch.sigmoid(m) * torch.sigmoid(-m)
        return (self.X.T * p.unsqueeze(0)) @ self.X / self.n

    def stoch_grad(self, w, rng):
        if self.batch_size is None or self.batch_size >= self.n:
            return self.grad_f(w)
        idx = torch.randint(0, self.n, (self.batch_size,), generator=rng, device=self.X.device)
        return self.grad_f(w, self.X[idx], self.y[idx])


class NonconvexClf:
    """Невыпуклая гладкая классификация: f(w) = 1/n sum phi(m_i), phi(t) = 1 - tanh(t),
    m_i = y_i x_i^T w. Ограниченная, гладкая, невыпуклая (для Т1/Т3)."""

    def __init__(self, path=None, n=2000, d=60, seed=0, device="cpu", batch_size=None):
        if path is not None:
            from sklearn.datasets import load_svmlight_file
            X, y = load_svmlight_file(path)
            X = torch.tensor(X.toarray(), dtype=torch.float64, device=device)
            y0 = torch.tensor(y, dtype=torch.float64, device=device)
            uniq = torch.unique(y0)
            y = torch.where(y0 == uniq[0], -1.0, 1.0).to(torch.float64)
            self.name = "nc_" + path.split("/")[-1].split(".")[0]
        else:
            g = torch.Generator().manual_seed(seed)
            X = torch.randn(n, d, generator=g, dtype=torch.float64).to(device)
            wtrue = torch.randn(d, generator=g, dtype=torch.float64).to(device)
            y = torch.sign(X @ wtrue + 0.3 * torch.randn(n, generator=g, dtype=torch.float64).to(device))
            self.name = "nc_synth"
        self.X, self.y = X, y
        self.n, self.d = X.shape
        self.batch_size = batch_size
        sv = torch.linalg.matrix_norm(X, ord=2)
        # |phi''| = |2 tanh sech^2| <= 4/(3 sqrt(3)) < 0.77
        self.L_f = float(0.77 * sv ** 2 / self.n)
        self.mu_f = 0.0
        self.device = device

    def f(self, w):
        m = self.y * (self.X @ w)
        return (1 - torch.tanh(m)).mean()

    def grad_f(self, w, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        m = y * (X @ w)
        dphi = -(1 - torch.tanh(m) ** 2)  # phi'(m) = -sech^2(m)
        return (X.T @ (y * dphi)) / X.shape[0]

    def diag_hess_f(self, w, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        m = y * (X @ w)
        t = torch.tanh(m)
        d2 = 2 * t * (1 - t ** 2)  # phi''(m)
        return (X * X * d2.unsqueeze(1)).sum(0) / X.shape[0]

    def hess_f(self, w):
        m = self.y * (self.X @ w)
        t = torch.tanh(m)
        d2 = 2 * t * (1 - t ** 2)
        return (self.X.T * d2.unsqueeze(0)) @ self.X / self.n

    def stoch_grad(self, w, rng):
        if self.batch_size is None or self.batch_size >= self.n:
            return self.grad_f(w)
        idx = torch.randint(0, self.n, (self.batch_size,), generator=rng, device=self.X.device)
        return self.grad_f(w, self.X[idx], self.y[idx])
