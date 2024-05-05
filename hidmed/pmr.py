"""Implementation of the PMR estimator based on Theorem 3"""

import numpy as np

from .proximal_estimator_base import ProximalEstimatorBase
from .cross_fit_base import CrossFittingEstimatorBase


class ProximalMultiplyRobustBase(ProximalEstimatorBase):
    """PMR estimator based on Theorem 3"""

    def fit(self, fit_data, val_data):
        """Fit the PMR estimator"""
        # estimate treatment probability
        if self.setup in ["b", "c"]:
            treatment, treatment_params, _ = self.fit_treatment_probability(
                fit_data,
                val_data,
            )
            self.treatment = treatment
            self.params["treatment"] = treatment_params

        # fit bridge functions
        q_fn, q_params, _ = self.fit_bridge(fit_data, val_data, which="q", treatment_prob=treatment)
        self.q_fn = q_fn
        self.params["q"] = q_params

        h_fn, h_params, _ = self.fit_bridge(fit_data, val_data, which="h")
        self.h_fn = h_fn
        self.params["h"] = h_params

        # fit eta
        eta, eta_params, _ = self.fit_eta(h_fn, fit_data, val_data)
        self.eta = eta
        self.params["eta"] = eta_params

        return self

    def evaluate(self, eval_data):
        """Evaluate the PMR estimator pointwise on the evaluation data"""
        if self.setup == "a":
            # estimate psi1
            loc0 = eval_data.a[:, 0] == 0
            loc1 = eval_data.a[:, 0] == 1
            q1 = self.q_fn(np.hstack((eval_data.z[loc1], eval_data.x[loc1])))
            h1 = self.h_fn(np.hstack((eval_data.w[loc1], eval_data.x[loc1])))
            h0 = self.h_fn(np.hstack((eval_data.w[loc0], eval_data.x[loc0])))
            eta_eval = self.eta.predict(eval_data.x)

            res = np.zeros(eval_data.n)
            res[loc1] = q1 * (eval_data.y[loc1, 0] - h1) * eval_data.n / np.sum(loc1)
            res[loc0] = (h0 - eta_eval[loc0]) * eval_data.n / np.sum(loc0)
            res += eta_eval
            return res

        # estimate psi2
        p_treat = self.treatment.predict_proba(eval_data.x)
        h_eval = self.h_fn(np.hstack((eval_data.w, eval_data.x)))
        q_eval = self.q_fn(np.hstack((eval_data.z, eval_data.x)))
        loc0 = eval_data.a[:, 0] == 0

        res = np.zeros(eval_data.n)
        res[loc0] = (
            p_treat[loc0, 1]
            * (h_eval[loc0] - self.eta.predict(eval_data.x[loc0]))
            * eval_data.n
            / np.sum(loc0)
        )
        res += eval_data.a[:, 0] * q_eval * (eval_data.y[:, 0] - h_eval)
        res += eval_data.a[:, 0] * self.eta.predict(eval_data.x)
        return res


class ProximalMultiplyRobust(CrossFittingEstimatorBase):
    """PIPW estimator with cross-fitting"""

    base_estimator = ProximalMultiplyRobustBase
