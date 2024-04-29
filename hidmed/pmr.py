"""Implementation of the PMR estimator based on Theorem 3"""

import numpy as np

from .cross_fit import CrossFittingEstimator


class ProximalMultiplyRobust(CrossFittingEstimator):
    """PMR estimator based on Theorem 3"""

    def __init__(
        self,
        generalized_model=True,
        kernel_metric="rbf",
        folds=2,
        num_runs=200,
        n_jobs=1,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            generalized_model=generalized_model,
            kernel_metric=kernel_metric,
            folds=folds,
            num_runs=num_runs,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        for key in ["treatment", "h", "q", "eta"]:
            if key in kwargs:
                self.params[key] = kwargs[key]

    def estimate(self, fit_data, eval_data, val_data):
        """Implements the PMR estimator"""
        # estimate bridge functions
        q_fn, q_params, _ = self.fit_bridge(fit_data, val_data, which="q")
        h_fn, h_params, _ = self.fit_bridge(fit_data, val_data, which="h")

        # estimate eta: E[h|A=0, X]
        eta, eta_params, _ = self.fit_eta(h_fn, fit_data, val_data)

        # save chosen parameters
        self.params["h"] = h_params
        self.params["q"] = q_params
        self.params["eta"] = eta_params

        # estimate treatment probability
        treatment_prob, treatment_params, _ = self.fit_treatment_probability(
            fit_data,
            val_data,
        )
        self.params["treatment"] = treatment_params

        # estimate psi2
        if self.generalized_model:
            p_treat = treatment_prob.predict_proba(eval_data.x)
            h_eval = h_fn(np.hstack((eval_data.w, eval_data.x)))
            q_eval = q_fn(np.hstack((eval_data.z, eval_data.x)))
            loc_0 = eval_data.a[:, 0] == 0

            res = np.zeros(eval_data.n)
            res[loc_0] = (
                p_treat[loc_0, 1]
                * (h_eval[loc_0] - eta.predict(eval_data.x[loc_0]))
                * eval_data.n
                / np.sum(loc_0)
            )
            res += eval_data.a[:, 0] * q_eval * (eval_data.y[:, 0] - h_eval)
            res += eval_data.a[:, 0] * eta.predict(eval_data.x)
            return res

        # estimate psi1
        loc_0 = eval_data.a[:, 0] == 0
        loc_1 = eval_data.a[:, 0] == 1
        q1 = q_fn(np.hstack((eval_data.z[loc_1], eval_data.x[loc_1])))
        h1 = h_fn(np.hstack((eval_data.w[loc_1], eval_data.x[loc_1])))
        h0 = h_fn(np.hstack((eval_data.w[loc_0], eval_data.x[loc_0])))
        eta_eval = eta.predict(eval_data.x)

        res = np.zeros(eval_data.n)
        res[loc_1] = q1 * (eval_data.y[loc_1, 0] - h1) * eval_data.n / np.sum(loc_1)
        res[loc_0] = (h0 - eta_eval[loc_0]) * eval_data.n / np.sum(loc_0)
        res += eta_eval
        return res
