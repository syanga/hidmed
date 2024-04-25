"""Estimator for the bridge function h(W, X)"""

import numpy as np

from .minimax import kkt_solve, score_nuisance_function
from .bridge_base import KernelBridgeBase


class KernelBridgeH(KernelBridgeBase):
    """Estimator for the bridge function h(W, X)"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        loc = data.a[:, 0] == 1
        g2 = data.y[loc, 0]
        g1 = -np.ones(len(g2))
        wx = np.hstack((data.w, data.x))
        zx = np.hstack((data.z, data.x))
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx, loc = self.extract_data(fit_data)

        kh1 = self.call_kernel(wx[loc], wx)
        kf1 = self.call_kernel(zx[loc], zx)
        kh = self.call_kernel(wx, wx)
        kf = self.call_kernel(zx, zx)

        alpha, _ = kkt_solve(
            kh1, kf1, kf1, kf1, kh, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = wx

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx, loc = self.extract_data(val_data)

        kf1 = self.call_kernel(zx[loc], zx)
        kf = self.call_kernel(zx, zx)

        try:
            return score_nuisance_function(
                self(wx[loc]),
                kf1,
                kf1,
                kf1,
                kf,
                g1,
                g2,
                self.lambda2,
            )
        except np.linalg.LinAlgError:
            return -np.inf
