import numpy as np

from .cross_fit import CrossFittingEstimator


class ProximalInverseProbWeighting(CrossFittingEstimator):
    """PIPW estimator based on Theorem 2"""

    def __init__(
        self,
        generalized_model=True,
        kernel_metric="rbf",
        folds=2,
        num_runs=200,
        n_jobs=1,
        verbose=True,
        q=None,
    ):
        super().__init__(
            generalized_model=generalized_model,
            kernel_metric=kernel_metric,
            folds=folds,
            num_runs=num_runs,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if q is not None:
            self.params["q"] = q

    def psi1_regression(self, q_fn, fit_data, val_data):
        """Fit conditional mean of y * q, given A=1, X"""

        def extract_data(data):
            loc = data.a[:, 0] == 1
            x0 = data.x[loc]
            targets = data.y[loc, 0] * q_fn(np.hstack((data.z[loc], data.x[loc])))
            return x0, targets

        x_fit, y_fit = extract_data(fit_data)
        x_val, y_val = extract_data(val_data)

        return self.fit_conditional_mean(
            x_fit, y_fit, x_val, y_val, name="psi1_regression"
        )

    def estimate(self, fit_data, eval_data, val_data):
        """Implements the PIPW estimator"""
        q_fn, q_params, _ = self.fit_bridge(fit_data, val_data, which="q")

        # save chosen parameters
        self.params["q"] = q_params

        # estimate psi_1
        if not self.generalized_model:
            loc_1 = eval_data.a[:, 0] == 1
            return np.mean(
                eval_data.y[loc_1, 0]
                * q_fn(np.hstack((eval_data.z[loc_1], eval_data.x[loc_1])))
            )

        # estimate for psi_2
        return np.mean(
            eval_data.a * eval_data.y * q_fn(np.hstack((eval_data.z, eval_data.x)))
        )
