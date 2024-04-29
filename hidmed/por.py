import numpy as np

from .cross_fit import CrossFittingEstimator


class ProximalOutcomeRegression(CrossFittingEstimator):
    """POR estimator based on Theorem 1"""

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

        # set up any provided parameters
        for key in ["treatment", "h", "eta"]:
            if key in kwargs:
                self.params[key] = kwargs[key]

    def estimate(self, fit_data, eval_data, val_data):
        """Implements the POR estimator"""
        # estimate treatment probability
        if self.generalized_model:
            treatment_prob, treatment_params, _ = self.fit_treatment_probability(
                fit_data,
                val_data,
            )

            # save chosen parameters
            self.params["treatment"] = treatment_params

            # psi2
            scale = treatment_prob.predict_proba(eval_data.x)[:, 1]
        else:
            # psi1
            scale = 1.0

        # fit bridge function h
        h_fn, h_params, _ = self.fit_bridge(fit_data, val_data, which="h")

        # fit conditional mean of h
        eta, eta_params, _ = self.fit_eta(h_fn, fit_data, val_data)

        # save chosen parameters
        self.params["h"] = h_params
        self.params["eta"] = eta_params

        # return estimate without treatment probability
        return eta.predict(eval_data.x) * scale
