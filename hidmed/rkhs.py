import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


""" create kernel function """


def create_kernel(metric="rbf", n_jobs=1, **kwds):
    def K(X, Y=None):
        return pairwise_kernels(X, Y, metric=metric, n_jobs=n_jobs, **kwds)

    return K


""" solve min max E[f(h*g1 + g2) - f^2] - lambda_f|f|^2 + lambda_h|h|^2 """


def solve_kkt(Kh, Kf, g1, g2, lambda_h, lambda_f):
    n = Kh.shape[0]
    assert (n, n) == Kh.shape
    assert (n, n) == Kf.shape
    A = np.block(
        [
            [2 * lambda_h * np.eye(n), (g1 * Kf.T).T / n],
            [-(g1 * Kh.T).T / n, 2 * (Kf / n + lambda_f * np.eye(n))],
        ]
    )
    b = np.hstack([np.zeros(n), g2 / n])
    sol = np.linalg.solve(A, b)
    alpha = sol[:n]
    beta = sol[n : 2 * n]
    return alpha, beta


""" fix h, solve max E[f(h*g1 + g2) - f^2] - lambda_f|f|^2 """


def solve_maxf(Kf, g1, g2, h, lambda_f):
    n = Kf.shape[0]
    assert (n, n) == Kf.shape
    A = Kf + n * lambda_f * np.eye(n)
    b = 0.5 * (h * g1 + g2)
    beta = np.linalg.solve(A, b)
    return beta
