import numpy as np
from .base import Method


class Newton(Method):
    """Демпфированный метод Ньютона с регуляризацией Левенберга и поиском Армихо.

    d_k = -(H_k + lambda I)^{-1} grad f(x_k),  lambda >= 0 — минимальный сдвиг,
          при котором H_k + lambda I положительно определена (нужно, если задача
          невыпукла в текущей точке);
    x_{k+1} = x_k + t_k d_k,  t_k — бэктрекинг по условию Армихо.

    Рядом с минимумом t_k = 1, lambda = 0 и сходимость квадратичная.
    Цена итерации — вычисление гессиана и решение системы n x n: O(n^3).
    """

    def __init__(self, oracle, alpha=0.5, beta_arm=1e-4, max_iter=100, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        self.alpha = alpha
        self.beta_arm = beta_arm
        self.name = "newton"

    def _direction(self, g, H):
        n = len(g)
        lam = 0.0
        scale = np.abs(np.diag(H)).max() + 1e-12
        while True:
            try:
                L = np.linalg.cholesky(H + lam * np.eye(n))
                break
            except np.linalg.LinAlgError:
                lam = max(2 * lam, 1e-6 * scale)
        return -np.linalg.solve(L.T, np.linalg.solve(L, g)), lam

    def _step(self, state):
        x = state["x"]
        f = self.oracle.f(x)
        g = self.oracle.grad(x)
        H = self.oracle.hess(x)
        d, lam = self._direction(g, H)
        t = 1.0
        slope = float(g @ d)
        for _ in range(60):
            if self.oracle.f(x + t * d) <= f + self.beta_arm * t * slope:
                break
            t *= self.alpha
        state["x_prev"] = x
        state["x"] = x + t * d
        state["step"] = t
        state["lambda"] = lam
        return state
