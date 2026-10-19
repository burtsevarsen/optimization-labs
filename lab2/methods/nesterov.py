import numpy as np
from .base import Method


class Nesterov(Method):
    """Ускоренный градиентный метод Нестерова.

    y_k     = x_k + beta_k (x_k - x_{k-1})
    x_{k+1} = y_k - eta * grad f(y_k)

    momentum="strongly_convex": beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)  (нужны L и mu)
    momentum="convex":          beta_k = (k - 1) / (k + 2)                   (нужен только L)

    restart=True — адаптивный рестарт (O'Donoghue, Candès 2015): если f выросла,
    импульс обнуляется. Полезно, когда mu неизвестна или задача невыпукла вдали от минимума.
    """

    def __init__(self, oracle, step=None, momentum="strongly_convex",
                 restart=True, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        info = oracle.info
        if step is None:
            if info.L is None:
                raise ValueError("нужен step или oracle.info.L")
            step = 1.0 / info.L
        self.step = float(step)
        self.momentum = momentum
        self.restart = restart
        if momentum == "strongly_convex":
            if info.L is None or info.mu is None:
                raise ValueError("для strongly_convex нужны oracle.info.L и mu")
            q = np.sqrt(info.mu / info.L)
            self._beta_const = (1 - q) / (1 + q)
        self.name = "nesterov" + ("+restart" if restart else "")
        self._t = 0          # счётчик с последнего рестарта

    def _beta(self):
        if self.momentum == "strongly_convex":
            return self._beta_const
        k = self._t
        return (k - 1) / (k + 2) if k > 0 else 0.0

    def _step(self, state):
        x, x_prev = state["x"], state["x_prev"]
        beta = self._beta()
        y = x + beta * (x - x_prev)
        g = self.oracle.grad(y)
        x_new = y - self.step * g
        self._t += 1
        if self.restart and self.oracle.f(x_new) > self.oracle.f(x):
            # рестарт: обычный шаг GD из x, импульс сбрасывается
            x_new = x - self.step * self.oracle.grad(x)
            self._t = 0
        state["x_prev"] = x
        state["x"] = x_new
        state["step"] = self.step
        return state
