import numpy as np
from .base import Method


class HeavyBall(Method):
    """Метод тяжёлого шара (Поляк, 1964).

    x_{k+1} = x_k - eta * grad f(x_k) + beta * (x_k - x_{k-1})

    Для сильно выпуклой квадратичной задачи с константами mu, L оптимальные параметры
        eta  = 4 / (sqrt(L) + sqrt(mu))^2
        beta = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^2,   kappa = L / mu
    и скорость сходимости ~ (1 - 1/sqrt(kappa)) вместо (1 - 1/kappa) у GD.
    """

    def __init__(self, oracle, step=None, beta=None, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        info = oracle.info
        L, mu = info.L, info.mu
        if step is None or beta is None:
            if L is None or mu is None:
                raise ValueError("нужны step и beta или oracle.info.L, mu")
            sL, smu = np.sqrt(L), np.sqrt(mu)
            step = 4.0 / (sL + smu) ** 2 if step is None else step
            beta = ((sL - smu) / (sL + smu)) ** 2 if beta is None else beta
        self.step = float(step)
        self.beta = float(beta)
        self.name = "heavy_ball"

    def _step(self, state):
        x, x_prev = state["x"], state["x_prev"]
        g = self.oracle.grad(x)
        x_new = x - self.step * g + self.beta * (x - x_prev)
        state["x_prev"] = x
        state["x"] = x_new
        state["step"] = self.step
        return state
