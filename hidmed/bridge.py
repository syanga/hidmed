"""Estimate bridge functions in the Proximal generalized hidden mediation model
using minimax optimization and kernel methods"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from .minimax import kkt_solve, score_nuisance_function


class BridgeEstimator:
    """Estimator for the bridge function in the Proximal generalized hidden
    mediation model"""

    def __init__(
        self,
        g1,
        g2,
        lambda1,
        lambda2,
        metric="rbf",
        gamma=1.0,
        degree=3.0,
        coef0=1.0,
        n_jobs=1,
    ):
        # minimax data
        self.g1 = g1.flatten()
        self.g2 = g2.flatten()

        # minimax regularization parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # kernel parameters
        self.kernel_args = {}
        if metric is not None:
            self.kernel_args["metric"] = metric
        if gamma is not None:
            self.kernel_args["gamma"] = gamma
        if degree is not None:
            self.kernel_args["degree"] = degree
        if coef0 is not None:
            self.kernel_args["coef0"] = coef0
        if n_jobs is not None:
            self.kernel_args["n_jobs"] = n_jobs

        # fitted values
        self.alpha = None
        self.x = None
        self.gram = None

    def call_kernel(self, x, y):
        """Call the kernel function with the given data"""
        return pairwise_kernels(x, y, filter_params=True, **self.kernel_args)

    def __call__(self, x):
        """Evaluate the bridge function at the given points"""
        return self.alpha @ self.call_kernel(self.x, x)

    @staticmethod
    def check_kernel_params(metric, gamma, degree, coef0):
        """Check if the kernel parameters are valid"""
        if metric is None or metric in ["linear", "additive_chi2", "cosine"]:
            if gamma is not None or degree is not None or coef0 is not None:
                return False

        elif metric in ["chi2", "rbf", "laplacian"]:
            if degree is not None or coef0 is not None:
                return False

        elif metric == "sigmoid":
            if degree is not None:
                return False

        return True

    def fit(self, xh, xf, yh1=None, yf0=None, yf1=None, yf2=None):
        """Fit the bridge function by solving the corresponding minimax
        problem"""
        kh = self.call_kernel(xh, xh)
        kf = self.call_kernel(xf, xf)
        kh1 = kh if yh1 is None else self.call_kernel(yh1, xh)
        kf0 = kf if yf0 is None else self.call_kernel(yf0, xf)
        kf1 = kf if yf1 is None else self.call_kernel(yf1, xf)
        kf2 = kf if yf2 is None else self.call_kernel(yf2, xf)
        self.alpha, _ = kkt_solve(
            kh1,
            kf0,
            kf1,
            kf2,
            kh,
            kf,
            self.g1,
            self.g2,
            self.lambda1,
            self.lambda2,
        )
        self.x = xh
        self.gram = kh1

    def score(self, xh, xf, yf0=None, yf1=None):
        """Score the fitted bridge function on validation data"""
        kf = self.call_kernel(x2, x2)
        kf0 = kf if y2_0 is None else self.call_kernel(y2_0, x2)
        kf1 = kf if y2_1 is None else self.call_kernel(y2_1, x2)
        try:
            return score_nuisance_function(
                self(x1),
                kf0,
                kf1,
                kf,
                self.g1,
                self.g2,
                self.lambda2,
            )
        except np.linalg.LinAlgError:
            return -np.inf
