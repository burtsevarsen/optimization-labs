"""Оракул задачи ценообразования при известном спросе (тема 6).

Модель
------
n товаров, вектор цен p > 0.
Спрос — изоэластичный с перекрёстными эффектами:

    d_i(p) = a_i * p_i^(-eps_i) * prod_{j != i} p_j^(gamma_ij)

    a_i      > 0  — базовый объём,
    eps_i    > 1  — собственная эластичность (на сколько % падает спрос при росте своей цены на 1 %),
    gamma_ij >= 0 — перекрёстная эластичность (товары-заменители: дорожает j — растёт спрос на i).

Прибыль: pi(p) = sum_i (p_i - c_i) d_i(p), c_i — себестоимость.

Минимизируем
    f(p) = -pi(p) + rho/2 * sum_i (log p_i - log p0_i)^2 + alpha/2 * ||p - p0||^2

    rho-слагаемое   — мягкое ограничение на ОТНОСИТЕЛЬНОЕ изменение цены (+ барьер p -> 0),
    alpha-слагаемое — мягкое ограничение на АБСОЛЮТНОЕ изменение цены.

Параметризации
--------------
param="price": переменная x = p (исходная постановка);
param="log":   переменная x = u = log p (замена переменных для повышения обусловленности).

Вся математика считается в u: log d = log a + G u, где G = -diag(eps) + Gamma.
Тогда выручка R_i = p_i d_i = a_i exp((I+G)_i u), затраты C_i = c_i d_i = c_i a_i exp(G_i u),
    pi(u)       = sum R_i - sum C_i
    grad_u pi   = (I+G)^T R - G^T C
    hess_u pi   = (I+G)^T diag(R) (I+G) - G^T diag(C) G
Для param="price" градиент и гессиан пересчитываются по цепному правилу (p = e^u):
    grad_p f = grad_u f / p
    hess_p f = diag(1/p) [hess_u f - diag(grad_u f)] diag(1/p)
"""
import numpy as np
from .base import Oracle, OracleInfo


class PricingOracle(Oracle):

    def __init__(self, a, eps, gamma, c, p0, rho: float = 0.0,
                 alpha: float = 0.0, param: str = "price",
                 names=None, scale: float = 1.0):
        self.a = np.asarray(a, dtype=float)
        self.eps = np.asarray(eps, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.rho = float(rho)
        self.alpha = float(alpha)
        if param not in ("price", "log"):
            raise ValueError("param must be 'price' or 'log'")
        self.param = param
        self.scale = float(scale)        # f делится на scale (нормировка масштаба)
        self.dim = len(self.a)
        self.names = list(names) if names is not None else [f"товар {i+1}" for i in range(self.dim)]
        self.u0 = np.log(self.p0)

        g = self.gamma.copy()
        np.fill_diagonal(g, 0.0)
        self.G = g - np.diag(self.eps)            # log d = log a + G u
        self.IG = np.eye(self.dim) + self.G       # log R = log a + (I+G) u
        self.log_a = np.log(self.a)

        self.name = f"pricing(n={self.dim}, rho={self.rho:g}, alpha={self.alpha:g}, param={param})"
        self._x_star = None
        self._f_star = None

    # ------------------------------------------------------------------
    # экономические величины
    # ------------------------------------------------------------------
    def demand(self, p):
        u = np.log(np.asarray(p, dtype=float))
        return np.exp(self.log_a + self.G @ u)

    def profit(self, p):
        p = np.asarray(p, dtype=float)
        return float(np.sum((p - self.c) * self.demand(p)))

    def to_price(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(x) if self.param == "log" else x

    def from_price(self, p):
        p = np.asarray(p, dtype=float)
        return np.log(p) if self.param == "log" else p

    # ------------------------------------------------------------------
    # ядро: значение, градиент, гессиан по u (без деления на scale)
    # ------------------------------------------------------------------
    def _parts_u(self, u, order):
        R = np.exp(self.log_a + self.IG @ u)            # выручка по товарам
        C = self.c * np.exp(self.log_a + self.G @ u)    # затраты по товарам
        p = np.exp(u)
        du = u - self.u0
        dp = p - self.p0

        f = -(R.sum() - C.sum()) + 0.5 * self.rho * du @ du + 0.5 * self.alpha * dp @ dp
        if order == 0:
            return f, None, None

        g = -(self.IG.T @ R - self.G.T @ C) + self.rho * du + self.alpha * dp * p
        if order == 1:
            return f, g, None

        H = -(self.IG.T @ (R[:, None] * self.IG) - self.G.T @ (C[:, None] * self.G))
        H = H + self.rho * np.eye(self.dim) + np.diag(self.alpha * (2 * p * p - self.p0 * p))
        return f, g, H

    def _eval(self, x, order):
        x = np.asarray(x, dtype=float)
        if self.param == "log":
            f, g, H = self._parts_u(x, order)
        else:
            if np.any(x <= 0):                   # цены вне области определения
                n = self.dim
                return np.inf, np.full(n, np.nan), np.full((n, n), np.nan)
            u = np.log(x)
            f, gu, Hu = self._parts_u(u, order)
            g = None if gu is None else gu / x
            H = None
            if Hu is not None:
                inv = 1.0 / x
                H = inv[:, None] * (Hu - np.diag(gu)) * inv[None, :]
        s = self.scale
        return f / s, (None if g is None else g / s), (None if H is None else H / s)

    # ------------------------------------------------------------------
    # интерфейс Oracle
    # ------------------------------------------------------------------
    def f(self, x):
        return float(self._eval(x, 0)[0])

    def grad(self, x):
        return self._eval(x, 1)[1]

    def hess(self, x):
        return self._eval(x, 2)[2]

    def hess_diag(self, x):
        return np.diag(self.hess(x))

    def hess_vec(self, x, v):
        return self.hess(x) @ v

    # ------------------------------------------------------------------
    # эталонное решение и обусловленность
    # ------------------------------------------------------------------
    def compute_reference(self, x0=None, tol=1e-12, max_iter=200):
        """Точный минимум демпфированным методом Ньютона (по u, затем перевод в текущую параметризацию).

        Нужен для метрик f - f*, ||x - x*||. Возвращает цены p*.
        """
        u = np.log(self.p0) if x0 is None else np.log(self.to_price(x0))
        for _ in range(max_iter):
            f, g, H = self._parts_u(u, 2)
            if np.linalg.norm(g) < tol * max(1.0, abs(f)):
                break
            # если гессиан не положительно определён — сдвиг (Левенберг)
            lam = 0.0
            while True:
                try:
                    L = np.linalg.cholesky(H + lam * np.eye(self.dim))
                    break
                except np.linalg.LinAlgError:
                    lam = max(2 * lam, 1e-8 * abs(f) + 1e-12)
            d = -np.linalg.solve(L.T, np.linalg.solve(L, g))
            t = 1.0
            while self._parts_u(u + t * d, 0)[0] > f + 1e-4 * t * g @ d and t > 1e-12:
                t *= 0.5
            u = u + t * d
            if not np.all(np.isfinite(u)) or np.abs(u - self.u0).max() > 20:
                raise RuntimeError("минимум не найден: цены уходят в 0 или бесконечность "
                                   "(задача неограничена — увеличьте rho/alpha)")
        p_star = np.exp(u)
        self._x_star = self.from_price(p_star)
        self._f_star = self.f(self._x_star)
        return p_star

    def condition_number(self, x):
        """Локальное число обусловленности: lambda_max / lambda_min гессиана (inf, если не PD)."""
        e = np.linalg.eigvalsh(self.hess(x))
        return float(e.max() / e.min()) if e.min() > 0 else float("inf")

    @property
    def info(self) -> OracleInfo:
        L = mu = None
        if self._x_star is not None:
            e = np.linalg.eigvalsh(self.hess(self._x_star))
            L, mu = float(e.max()), float(e.min())
        return OracleInfo(L=L, mu=mu, x_star=self._x_star, f_star=self._f_star)

    def with_param(self, param, scale=None):
        """Та же задача в другой параметризации (и/или масштабе)."""
        o = PricingOracle(self.a, self.eps, self.gamma, self.c, self.p0,
                          self.rho, self.alpha, param, self.names,
                          self.scale if scale is None else scale)
        if self._x_star is not None:
            o.compute_reference(o.from_price(self.to_price(self._x_star)))
        return o


# ----------------------------------------------------------------------
# Синтетические данные: линейка напитков в чайной
# ----------------------------------------------------------------------
TEA_MENU = [
    # название,                         группа,  цена p0 ₽, себест. c ₽, продажи q0 шт/день, eps
    # eps откалиброваны от «правила Лернера» p/(p-c) — эластичности, при которой текущая цена
    # была бы оптимальной для одиночного товара: eps > p/(p-c) — напиток переоценён, eps < — недооценён.
    ("Молочный чай с тапиокой",          "milk",  350,  90, 120, 1.41),
    ("Молочный чай таро",                "milk",  380, 100,  60, 1.70),
    ("Матча латте",                      "milk",  420, 125,  45, 1.54),
    ("Брауншугар с молоком",             "milk",  390, 105,  80, 1.85),
    ("Фруктовый чай манго",              "fruit", 340,  85,  90, 1.73),
    ("Фруктовый чай маракуйя",           "fruit", 340,  85,  70, 1.87),
    ("Фруктовый чай клубника",           "fruit", 330,  80,  65, 1.91),
    ("Чизфом улун",                      "other", 410, 115,  40, 1.46),
    ("Лимонад юдзу",                     "other", 300,  70,  50, 1.96),
    ("Холодный жасминовый чай",          "other", 250,  50,  55, 2.00),
]


def tea_shop(n: int = 10, gamma_in: float = 0.08, gamma_out: float = 0.02,
             rho: float = 0.0, alpha: float = 0.0, param: str = "price",
             scale: float = 1.0, seed: int | None = None, noise: float = 0.0,
             items=None, eps_override=None):
    """Создаёт PricingOracle для напитков меню (первые n или список индексов items).

    gamma_in  — перекрёстная эластичность внутри группы (молочные / фруктовые / прочие),
    gamma_out — между группами.
    a_i подбирается так, чтобы при текущих ценах p0 продажи были ровно q0 (калибровка).
    noise > 0 — случайный разброс eps и gamma (относительный), для экспериментов.
    """
    menu = [TEA_MENU[i] for i in items] if items is not None else TEA_MENU[:n]
    n = len(menu)
    names = [m[0] for m in menu]
    groups = [m[1] for m in menu]
    p0 = np.array([m[2] for m in menu], float)
    c = np.array([m[3] for m in menu], float)
    q0 = np.array([m[4] for m in menu], float)
    eps = np.array([m[5] for m in menu], float) if eps_override is None else np.asarray(eps_override, float)

    gamma = np.array([[0.0 if i == j else (gamma_in if groups[i] == groups[j] else gamma_out)
                       for j in range(n)] for i in range(n)])
    if noise > 0:
        rng = np.random.default_rng(seed)
        eps = eps * (1 + noise * rng.uniform(-1, 1, n))
        gamma = gamma * (1 + noise * rng.uniform(-1, 1, (n, n)))
        np.fill_diagonal(gamma, 0.0)

    # калибровка: d_i(p0) = q0_i  =>  log a = log q0 - G log p0
    G = gamma - np.diag(eps)
    a = np.exp(np.log(q0) - G @ np.log(p0))
    return PricingOracle(a, eps, gamma, c, p0, rho=rho, alpha=alpha,
                         param=param, names=names, scale=scale)


# ----------------------------------------------------------------------
# Робастификация: неопределённость в эластичностях
# ----------------------------------------------------------------------
class ScenarioPricingOracle(Oracle):
    """Целевая функция по набору сценариев эластичностей eps^(1..K).

    mode="mean":    f(x) = 1/K sum_k f_k(x)                      — ожидаемая (SAA) постановка;
    mode="softmax": f(x) = 1/tau * log( 1/K sum_k exp(tau f_k(x)) ) — гладкая аппроксимация
                    худшего случая max_k f_k(x) (tau -> inf: max, tau -> 0: mean).
    Все f_k — PricingOracle с одинаковыми c, p0, rho, alpha и параметризацией.
    """

    def __init__(self, oracles, mode="mean", tau=1.0):
        self.oracles = list(oracles)
        self.mode = mode
        self.tau = float(tau)
        o0 = self.oracles[0]
        self.dim, self.param, self.p0, self.u0 = o0.dim, o0.param, o0.p0, o0.u0
        self.names, self.c = o0.names, o0.c
        self.name = f"scenarios(K={len(self.oracles)}, mode={mode})"
        self._x_star = self._f_star = None

    def to_price(self, x):
        return self.oracles[0].to_price(x)

    def from_price(self, p):
        return self.oracles[0].from_price(p)

    def _weights(self, fs):
        if self.mode == "mean":
            return np.full(len(fs), 1.0 / len(fs))
        z = self.tau * (fs - fs.max())
        w = np.exp(z)
        return w / w.sum()

    def f(self, x):
        fs = np.array([o.f(x) for o in self.oracles])
        if self.mode == "mean":
            return float(fs.mean())
        m = fs.max()
        return float(m + np.log(np.mean(np.exp(self.tau * (fs - m)))) / self.tau)

    def grad(self, x):
        fs = np.array([o.f(x) for o in self.oracles])
        w = self._weights(fs)
        return sum(wk * o.grad(x) for wk, o in zip(w, self.oracles))

    def hess(self, x):
        fs = np.array([o.f(x) for o in self.oracles])
        w = self._weights(fs)
        gs = [o.grad(x) for o in self.oracles]
        H = sum(wk * o.hess(x) for wk, o in zip(w, self.oracles))
        if self.mode == "softmax":
            gbar = sum(wk * g for wk, g in zip(w, gs))
            H = H + self.tau * sum(wk * np.outer(g - gbar, g - gbar) for wk, g in zip(w, gs))
        return H

    def profits(self, p):
        """Прибыль в каждом сценарии при ценах p."""
        return np.array([o.profit(p) for o in self.oracles])

    def compute_reference(self, x0=None, tol=1e-10, max_iter=200):
        from methods.newton import Newton
        x0 = self.from_price(self.p0) if x0 is None else x0
        h = Newton(self, max_iter=max_iter, tol=tol).run(x0)
        self._x_star = h.x[-1]
        self._f_star = self.f(self._x_star)
        return self.to_price(self._x_star)

    @property
    def info(self) -> OracleInfo:
        L = mu = None
        if self._x_star is not None:
            e = np.linalg.eigvalsh(self.hess(self._x_star))
            L, mu = float(e.max()), float(e.min())
        return OracleInfo(L=L, mu=mu, x_star=self._x_star, f_star=self._f_star)


def tea_shop_scenarios(K: int = 50, sigma: float = 0.15, seed: int = 0, **kw):
    """K сценариев: eps_i умножаются на exp(sigma * N(0,1)) (ошибка оценки эластичности ~sigma).

    a_i каждый раз перекалибровывается, поэтому при текущих ценах продажи совпадают с q0:
    неопределённость только в том, КАК спрос реагирует на изменение цены.
    Сценарии с eps <= 1.05 отбрасываются (при eps <= 1 оптимальной цены нет).
    """
    rng = np.random.default_rng(seed)
    base = tea_shop(**kw)
    out = []
    while len(out) < K:
        eps = base.eps * np.exp(sigma * rng.standard_normal(base.dim))
        if np.all(eps > 1.05):
            out.append(tea_shop(eps_override=eps, **kw))
    return out
