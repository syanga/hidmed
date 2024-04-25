import numpy as np
from numba import njit


@njit
def kkt_solve(kh1, kf0, kf1, kf2, kh, kf, g1, g2, lambda1, lambda2):
    """
    The goal is to find kernel functions h(r1) (and function f(r2)) that solves:

    h = \argmin_{h} \max_f [
        \mathbb{E}(f(r2)*(h(r1)*g1 + g2) - f(r2)**2)
        - lambda_2 * \|f\|_{RKHS}^2 + lambda_1 *\|h\|_{RKHS}^2
    ]

    The function h is in the RKHS of the metric specified by the kernel function
    with gram matrix gram1, and the function f is in the RKHS of the metric
    specified by the kernel function with gram matrix gram2.
    """
    n = kh.shape[0]
    kkt_matrix = np.zeros((3 * n, 3 * n))
    kkt_matrix[:n, :n] = 2 * lambda1 * kh
    kkt_matrix[:n, n : 2 * n] = np.ascontiguousarray(kh1.T).dot(
        np.ascontiguousarray((g1 * kf1.T).T)
    )
    kkt_matrix[:n, 2 * n :] = kkt_matrix[:n, n : 2 * n]
    kkt_matrix[n : 2 * n, :n] = kkt_matrix[:n, n : 2 * n].T
    kkt_matrix[n : 2 * n, n : 2 * n] = -2 * (kf2.T.dot(kf2) + lambda2 * kf)
    kkt_matrix[2 * n :, 2 * n :] = kkt_matrix[n : 2 * n, n : 2 * n]

    kkt_vec = np.zeros(3 * n)
    kkt_vec[n : 2 * n] = -(kf0.T.dot(np.ascontiguousarray(g2)))

    sol = np.linalg.solve(kkt_matrix, kkt_vec)
    alpha, beta = sol[:n], sol[n : 2 * n]
    return alpha, beta


@njit
def score_nuisance_function(h1, kf0, kf1, kf2, kf, g1, g2, lambda2):
    """
    Score a fitted nuisance function (function h) by solving the minimax problem
    with respect to the beta vector (function f), with the values h(r1) fixed.
    Returns a score (higher is better).
    """
    # form full KKT system
    kkt_matrix = kf2.T.dot(kf2) + lambda2 * kf
    kkt_vector = (kf0.T.dot(g2) + kf1.T.dot(g1 * h1)) * 0.5
    beta = np.linalg.solve(kkt_matrix, kkt_vector)
    f_values_0 = kf0.dot(beta)
    f_values_1 = kf1.dot(beta)
    f_values_2 = kf2.dot(beta)

    metric = (
        np.mean(g1 * h1 * f_values_1)
        + np.mean(g2 * f_values_0)
        - np.mean(f_values_2**2)
    )
    return -metric
