"""Estimator for the bridge function h(W, X)"""

import numpy as np

from .minimax import kkt_solve, score_nuisance_function
from .bridge_base import KernelBridgeBase


class KernelBridgeQProp(KernelBridgeBase):
    """Estimator for the bridge function Q(Z, X) using propensity score weighting"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        treatment_probs = self.treatment_prob.predict_proba(data.x)
        g1 = data.a[:, 0] / treatment_probs[:, 1]
        g2 = -(1.0 - data.a[:, 0]) / treatment_probs[:, 0]
        zx = np.hstack((data.z, data.x))
        wx = np.hstack((data.w, data.x))
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.call_kernel(zx, zx)
        kf = self.call_kernel(wx, wx)

        alpha, _ = kkt_solve(kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2)

        self.alpha = alpha
        self.x = zx

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx = self.extract_data(val_data)

        q1 = self(zx)
        kf = self.call_kernel(wx, wx)

        try:
            return score_nuisance_function(
                q1,
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
                self.lambda2,
            )
        except np.linalg.LinAlgError:
            return -np.inf
