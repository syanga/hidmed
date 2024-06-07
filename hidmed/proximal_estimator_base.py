"""Base class for cross-fitting estimators."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

from hola.tune import tune

from .bridge_h import KernelBridgeH
from .bridge_q import KernelBridgeQ

from .parameters import (
    LAMBDA_MIN_FACTOR,
    LAMBDA_MAX_FACTOR,
    LAMBDA_GRID,
    GAMMA_MIN,
    GAMMA_MAX,
    GAMMA_GRID,
    GAMMA_VALUE,
    REG_GAMMA_MIN,
    REG_GAMMA_MAX,
    ALPHA_MIN,
    ALPHA_MAX,
    C_MIN,
    C_MAX,
    DEGREE_MIN,
    DEGREE_MAX,
)


class ProximalEstimatorBase:

    def __init__(
        self,
        setup,
        kernel_metric="rbf",
        num_runs=200,
        num_jobs=1,
        verbose=True,
        **kwargs,
    ):
        assert setup in ["a", "b", "c"], "Invalid setup. Choose from ['a', 'b', 'c']"
        assert kernel_metric in [
            "rbf",
            "laplacian",
        ], "Invalid kernel metric. Choose from ['rbf', 'laplacian']"
        self.setup = setup
        self.kernel_metric = kernel_metric
        self.num_runs = num_runs
        self.num_jobs = num_jobs
        self.verbose = verbose

        # set any provided parameters
        self.params = {}
        for key in ["treatment", "h", "q", "eta"]:
            if key in kwargs:
                self.params[key] = kwargs[key]

        # pass in dgp directly for debug
        self.dgp = kwargs.get("dgp", None)

    def fit(self, fit_data, val_data):
        """Fit the estimator"""
        raise NotImplementedError

    def evaluate(self, eval_data):
        """Evaluate the estimator after fitting pointwise on the evaluation data"""
        raise NotImplementedError

    def fit_bridge(self, fit_data, val_data, which="h", treatment_prob=None):
        """Fit bridge function with hyperparameter tuning"""
        if treatment_prob is None and which == "q":
            # fit propensity score if needed
            treatment_prob, treatment_params, _ = self.fit_treatment_probability(fit_data, val_data)
            self.treatment = treatment_prob
            self.params["treatment"] = treatment_params

        # which function to fit
        method = KernelBridgeQ if which == "q" else KernelBridgeH

#        def _fit_bridge(lambda1, lambda2, gamma1, gamma2):
#            est = method(lambda1, lambda2, gamma1, gamma2, treatment_prob=treatment_prob)
#            est.fit(fit_data)
#            return {"score": est.score(val_data)}

        def _fit_bridge(lambda1, lambda2, gamma):
            est = method(lambda1, lambda2, gamma, gamma, treatment_prob=treatment_prob)
            est.fit(fit_data)
            return {"score": est.score(val_data)}

        # set up hyperparameter tuning
        params_config = {}
        for reg_param in ["lambda1", "lambda2"]:
            if (
                    self.params.get(which, None) is not None
                    and self.params[which].get(reg_param, None) is not None
            ):
                params_config[reg_param] = {"values": [self.params[which][reg_param]]}
            else:
                params_config[reg_param] = {
                    "min": LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    "max": LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2,
                    "scale": "log",
                    # "grid": LAMBDA_GRID,
                }

#        for bandwidth_param in ["gamma1", "gamma2"]:
        for bandwidth_param in ["gamma"]:
            if (
                    self.params.get(which, None) is not None
                    and self.params[which].get(bandwidth_param, None) is not None
            ):
                params_config[bandwidth_param] = {"values": [self.params[which][bandwidth_param]]}
            else:
                params_config[bandwidth_param] = {
                    "min": GAMMA_MIN,
                    "max": GAMMA_MAX,
                    "scale": "log",
                    # "grid": GAMMA_GRID,
                }

        # hyperparameter tuning, or use provided values
        if any([len(v.get("values", [])) != 1 for _, v in params_config.items()]):
            objectives_config = {"score": {"target": 0.0, "limit": 10.0}}
            tuner = tune(
                _fit_bridge,
                params_config,
                objectives_config,
                self.num_runs,
                self.num_jobs,
            )
            params = tuner.get_best_params()
            scores = tuner.get_best_scores()
        else:
            params = self.params[which]
            scores = {"score": np.nan}

        # fit bridge function
        bridge = method(
            params["lambda1"],
            params["lambda2"],
#            params["gamma1"],
#            params["gamma2"],
            params["gamma"],
            params["gamma"],
            treatment_prob=treatment_prob,
        )
        bridge.fit(fit_data)
        if self.verbose:
            print(f"Bridge {which} params: {params}, score: {scores['score']}")

        return bridge, params, scores

    def fit_eta(self, h_fn, fit_data, val_data):
        """Fit conditional mean of h, given A=0, X"""

        def extract_data(data):
            loc = data.a[:, 0] == 0
            x0 = data.x[loc]
            targets = h_fn(np.hstack((data.w[loc], data.x[loc])))
            return x0, targets

        x_fit, y_fit = extract_data(fit_data)
        x_val, y_val = extract_data(val_data)

        return self.fit_conditional_mean(x_fit, y_fit, x_val, y_val, name="eta")

    def fit_conditional_mean(self, X, y, X_val, y_val, name="conditional_mean"):
        """Fit kernel ridge regression with hyperparameter tuning"""
        def build_reg(alpha, gamma):
            return Pipeline(
                [
                    # ("scaler", StandardScaler()),
                    (
                        "kernel_ridge",
                        KernelRidge(
                            kernel=self.kernel_metric, alpha=alpha, gamma=gamma
                        ),
                    ),
                ]
            )

        def _fit_conditional_mean(alpha, gamma):
            reg = build_reg(alpha, gamma)
            reg.fit(X, y)
            return {"r2": reg.score(X_val, y_val)}

        params_config = {}
        if (
            self.params.get(name, None) is not None
            and self.params[name].get("alpha", None) is not None
        ):
            params_config["alpha"] = {"values": [self.params[name]["alpha"]]}
        else:
            params_config["alpha"] = {
                "min": ALPHA_MIN,
                "max": ALPHA_MAX,
                "scale": "log",
            }
        if (
            self.params.get(name, None) is not None
            and self.params[name].get("gamma", None) is not None
        ):
            params_config["gamma"] = {"values": [self.params[name]["gamma"]]}
        else:
            params_config["gamma"] = {
                "min": REG_GAMMA_MIN,
                "max": REG_GAMMA_MAX,
                "scale": "log",
            }

        # hyperparameter tuning, or use provided values
        if any([len(v.get("values", [])) != 1 for _, v in params_config.items()]):
            objectives_config = {
                "r2": {"target": 1.0, "limit": 0.0},
            }
            tuner = tune(
                _fit_conditional_mean,
                params_config,
                objectives_config,
                max(200, self.num_runs),
                self.num_jobs,
            )

            params = tuner.get_best_params()
            scores = tuner.get_best_scores()
        else:
            params = self.params[name]
            scores = {"r2": np.nan}

        # fit conditional mean
        cond_mean = build_reg(params["alpha"], params["gamma"])
        cond_mean.fit(X, y)

        if self.verbose:
            print(f"{name} params: {params}, r2: {scores['r2']}")

        return cond_mean, params, scores

    def fit_treatment_probability(self, fit_data, val_data):
        """Kernel logistic regression for treatment probability given
        covariates"""
        def build_reg(C, degree):
            return Pipeline(
                [
                    # ("std_scaler", StandardScaler()),
                    # ("poly", PolynomialFeatures(degree=int(degree))),
                    ("logistic", LogisticRegression(C=C)),
                ]
            )

        def fit_prob(C, degree):
            reg = build_reg(C, degree)
            reg.fit(fit_data.x, fit_data.a.flatten())
            return {
                "log_loss": log_loss(
                    val_data.a.flatten(), reg.predict_proba(val_data.x)
                )
            }

        params_config = {}
        if (
            self.params.get("treatment", None) is not None
            and self.params["treatment"].get("C", None) is not None
        ):
            params_config["C"] = {"values": [self.params["treatment"]["C"]]}
        else:
            params_config["C"] = {
                "min": C_MIN,
                "max": C_MAX,
                "scale": "log",
            }
        if (
            self.params.get("treatment", None) is not None
            and self.params["treatment"].get("degree", None) is not None
        ):
            params_config["degree"] = {"values": [self.params["treatment"]["degree"]]}
        else:
            params_config["degree"] = {
                "min": DEGREE_MIN,
                "max": DEGREE_MAX,
                "param_type": "int",
                "grid": DEGREE_MAX - DEGREE_MIN + 1,
            }

        # hyperparameter tuning, or use provided values
        if any([len(v.get("values", [])) != 1 for _, v in params_config.items()]):
            objectives_config = {"log_loss": {"target": 0.0, "limit": 1e1}}
            tuner = tune(
                fit_prob,
                params_config,
                objectives_config,
                max(200, self.num_runs),
                self.num_jobs,
            )

            params = tuner.get_best_params()
            scores = tuner.get_best_scores()
        else:
            params = self.params["treatment"]
            scores = {"log_loss": np.nan}

        # fit treatment probability
        prob = build_reg(params["C"], params["degree"])
        prob.fit(fit_data.x, fit_data.a.flatten())

        if self.verbose:
            print(
                "Treatment prob params:",
                {k: np.round(v, 3) for k, v in params.items()},
                "log_loss: ",
                np.round(scores["log_loss"], 3),
            )

        return prob, params, scores
