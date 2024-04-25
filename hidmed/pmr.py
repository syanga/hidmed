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
        treatment=None,
        h=None,
        q=None,
        eta=None,
    ):
        super().__init__(
            generalized_model=generalized_model,
            kernel_metric=kernel_metric,
            folds=folds,
            num_runs=num_runs,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if treatment is not None:
            self.params["treatment"] = treatment
        if h is not None:
            self.params["h"] = h
        if q is not None:
            self.params["q"] = q
        if eta is not None:
            self.params["eta"] = eta

    def estimate(self, fit_data, eval_data, val_data):
        """Implements the PMR estimator"""
        # estimate bridge functions
        h_fn, h_params, _ = self.fit_bridge(fit_data, val_data, which="h")

        # estimate treatment probability
        if self.generalized_model:
            treatment_prob, treatment_params, _ = self.fit_treatment_probability(
                fit_data,
                val_data,
            )
            self.params["treatment"] = treatment_params
        else:
            treatment_prob = None

        q_fn, q_params, _ = self.fit_bridge(
            fit_data, val_data, which="q", treatment_prob=treatment_prob
        )

        # estimate eta: E[h|A=0, X]
        eta, eta_params, _ = self.fit_eta(h_fn, fit_data, val_data)

        # save chosen parameters
        self.params["h"] = h_params
        self.params["q"] = q_params
        self.params["eta"] = eta_params

        # estimate psi2
        if self.generalized_model:
            p_treat = treatment_prob.predict_proba(eval_data.x)
            h_eval = h_fn(np.hstack((eval_data.w, eval_data.x)))
            q_eval = q_fn(np.hstack((eval_data.z, eval_data.x)))
            loc_0 = eval_data.a[:, 0] == 0
            psi2 = np.mean(eval_data.a[:, 0] * q_eval * (eval_data.y - h_eval))
            psi2 += np.mean(
                p_treat[loc_0, 1] * (h_eval[loc_0] - eta.predict(eval_data.x[loc_0]))
            )
            psi2 += np.mean(eval_data.a[:, 0] * eta.predict(eval_data.x))
            return psi2

        loc_0 = eval_data.a[:, 0] == 0
        loc_1 = eval_data.a[:, 0] == 1
        q1 = q_fn(np.hstack((eval_data.z[loc_1], eval_data.x[loc_1])))
        h1 = h_fn(np.hstack((eval_data.w[loc_1], eval_data.x[loc_1])))
        h0 = h_fn(np.hstack((eval_data.w[loc_0], eval_data.x[loc_0])))
        psi1 = np.mean(q1 * (eval_data.y[loc_1, 0] - h1))
        psi1 += np.mean(h0 - eta.predict(eval_data.x[loc_0]))
        psi1 += np.mean(eta.predict(eval_data.x))
        return psi1
