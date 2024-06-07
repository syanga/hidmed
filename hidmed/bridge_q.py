"""Estimator for the bridge function h(W, X)"""

import numpy as np

from .minimax import kkt_solve, score_nuisance_function
from .bridge_base import KernelBridgeBase
from .parameters import LAMBDA_MIN_FACTOR, LAMBDA_MAX_FACTOR


class KernelBridgeQ(KernelBridgeBase):
    """Estimator for the bridge function Q(Z, X)"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
#         loc_0 = data.a[:, 0] == 0
#         loc_1 = data.a[:, 0] == 1
#         n0 = np.sum(loc_0)
#         n1 = np.sum(loc_1)
#         g1 = np.ones(n1) * len(data) / n1
#         g2 = -1.0 * np.ones(n0) * len(data) / n0
#         zx = np.hstack((data.z, data.x))
#         wx = np.hstack((data.w, data.x))
#         return g1, g2, wx, zx, loc_0, loc_1

        probs = self.treatment_prob.predict_proba(data.x)
        g1 = (data.a[:, 0] == 1).astype(float) / probs[:, 1]
        g2 = -(data.a[:, 0] == 0).astype(float) / probs[:, 0]
        wx = np.hstack((data.w, data.x))
        zx = np.hstack((data.z, data.x))
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
#         g1, g2, wx, zx, loc_0, loc_1 = self.extract_data(fit_data)

#         kq1 = self.kernel1(zx[loc_1], zx)
#         kf0 = self.kernel2(wx[loc_0], wx)
#         kf1 = self.kernel2(wx[loc_1], wx)

#         kq = self.kernel1(zx, zx)
#         kf = self.kernel2(wx, wx)

#         alpha, beta = kkt_solve(
#             kq1, kf0, kf1, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
#         )

#         self.alpha = alpha
#         self.x = zx.copy()
#         self.beta = beta
#         self.xf = wx.copy()

        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.kernel1(zx, zx)
        kf = self.kernel2(wx, wx)

        alpha, beta = kkt_solve(
            kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = zx.copy()
        self.beta = beta
        self.xf = wx.copy()

    def score(self, val_data):
        """Score the bridge function"""
#         g1, g2, wx, zx, loc_0, loc_1 = self.extract_data(val_data)

#         q1 = self(zx[loc_1])
#         kf0 = self.kernel2(wx[loc_0], wx)
#         kf1 = self.kernel2(wx[loc_1], wx)
#         kf = self.kernel2(wx, wx)
#         try:
#             return score_nuisance_function(
#                 q1,
#                 kf0,
#                 kf1,
#                 kf,
#                 kf,
#                 g1,
#                 g2,
# #                 self.lambda2,
#                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
#             )
#         except np.linalg.LinAlgError:
#             return np.inf

        g1, g2, wx, zx = self.extract_data(val_data)

        kf = self.kernel2(wx, wx)
        try:
            return score_nuisance_function(
                self(zx),
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
#                 self.lambda2,
#                 np.sqrt(LAMBDA_MIN_FACTOR * LAMBDA_MAX_FACTOR) * kf.shape[0] ** 0.2,
                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf
