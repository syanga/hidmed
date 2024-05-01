import numpy as np

from .proximal_estimator_base import ProximalEstimatorBase
from .cross_fit_base import CrossFittingEstimatorBase


class ProximalInverseProbWeightingBase(ProximalEstimatorBase):
    """PIPW estimator based on Theorem 2"""

    def fit(self, fit_data, val_data):
        """Fit the PIPW estimator"""
        # fit q bridge function
        q_fn, q_params, _ = self.fit_bridge(fit_data, val_data, which="q")
        self.q_fn = q_fn
        self.params["q"] = q_params

        return self

    def evaluate(self, eval_data):
        """Evaluate the PIPW estimator pointwise on the evaluation data"""
        if self.setup == "a":
            loc1 = eval_data.a[:, 0] == 1
            q1 = self.q_fn(np.hstack((eval_data.z[loc1], eval_data.x[loc1])))
            return eval_data.y[loc1, 0] * q1

        q_eval = self.q_fn(np.hstack((eval_data.z, eval_data.x)))
        return eval_data.a[:, 0] * eval_data.y[:, 0] * q_eval


class ProximalInverseProbWeighting(CrossFittingEstimatorBase):
    """PIPW estimator with cross-fitting"""

    base_estimator = ProximalInverseProbWeightingBase
