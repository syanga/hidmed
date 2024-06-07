import numpy as np

from .proximal_estimator_base import ProximalEstimatorBase
from .cross_fit_base import CrossFittingEstimatorBase
from .parameters import MIN_PROP_SCORE


class ProximalOutcomeRegressionBase(ProximalEstimatorBase):
    """POR estimator based on Theorem 1"""

    def fit(self, fit_data, val_data):
        """Fit the POR estimator"""
        # estimate treatment probability
        if self.setup in ["b", "c"]:
            self.treatment, treatment_params, _ = self.fit_treatment_probability(
                fit_data,
                val_data,
            )
            self.params["treatment"] = treatment_params

        # fit bridge function h
        h_fn, h_params, _ = self.fit_bridge(fit_data, val_data, which="h")
        self.h_fn = h_fn
        self.params["h"] = h_params

        # fit conditional mean of h
        eta, eta_params, _ = self.fit_eta(h_fn, fit_data, val_data)
        self.eta = eta
        self.params["eta"] = eta_params

        return self

    def evaluate(self, eval_data):
        """Evaluate the POR estimator pointwise on the evaluation data"""
        # estimate treatment probability
        if self.setup in ["b", "c"]:
            treatment_prob = np.clip(self.treatment.predict_proba(eval_data.x)[:, 1], MIN_PROP_SCORE, 1-MIN_PROP_SCORE)
        else:
            treatment_prob = 1.0

        # return estimate without treatment probability
        return self.eta.predict(eval_data.x) * treatment_prob


class ProximalOutcomeRegression(CrossFittingEstimatorBase):
    """PIPW estimator with cross-fitting"""

    base_estimator = ProximalOutcomeRegressionBase
