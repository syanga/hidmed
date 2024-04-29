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


LAMBDA_MIN_FACTOR = 1e-6
LAMBDA_MAX_FACTOR = 50.0
LAMBDA_GRID = 20

GAMMA_MIN = 1e-3
GAMMA_MAX = 1e1
GAMMA_GRID = 50

REG_GAMMA_MIN = 1e-3
REG_GAMMA_MAX = 1e1

ALPHA_MIN = 1e-7
ALPHA_MAX = 1e1

C_MIN = 1e-1
C_MAX = 1e7

DEGREE_MIN = 1
DEGREE_MAX = 3


class CrossFittingEstimator:
    """Estimator that uses cross-fitting to estimate the nuisance functions"""

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
        # estimate psi2 if generalized model, else estimate psi1
        self.generalized_model = generalized_model

        # kernel hyperparameters
        assert kernel_metric in ["rbf", "laplacian"]
        self.kernel_metric = kernel_metric

        # cross-fitting folds
        self.folds = max(1, folds)

        # place to store provided or selected hyperparameters
        self.params = {}

        # hyperparameter tuning with HOLA
        self.num_runs = num_runs
        self.n_jobs = n_jobs

        # verbose output
        self.verbose = verbose

    def fit_bridge(self, fit_data, val_data, which="h"):
        """Fit bridge function with hyperparameter tuning"""
        if which == "h":
            method = KernelBridgeH
        else:
            method = KernelBridgeQ

        def _fit_bridge(lambda1, lambda2, gamma1, gamma2):
            est = method(lambda1, lambda2, gamma1, gamma2)
            est.fit(fit_data)
            score = est.score(val_data)
            return {
                "score": score,
                "gamma1": gamma1,
                "gamma2": gamma2,
                "lambda1": lambda1,
                "lambda2": lambda2,
            }

        # set up hyperparameter tuning
        params_config = {}
        if (
            self.params.get(which, None) is not None
            and self.params[which].get("lambda1", None) is not None
        ):
            params_config["lambda1"] = {"values": [self.params[which]["lambda1"]]}
        else:
            params_config["lambda1"] = {
                "min": LAMBDA_MIN_FACTOR * len(fit_data),
                "max": LAMBDA_MAX_FACTOR * len(fit_data),
                "scale": "log",
                "grid": LAMBDA_GRID,
            }
        if (
            self.params.get(which, None) is not None
            and self.params[which].get("lambda2", None) is not None
        ):
            params_config["lambda2"] = {"values": [self.params[which]["lambda2"]]}
        else:
            params_config["lambda2"] = {
                "min": LAMBDA_MIN_FACTOR * len(fit_data),
                "max": LAMBDA_MAX_FACTOR * len(fit_data),
                "scale": "log",
                "grid": LAMBDA_GRID,
            }
        if (
            self.params.get(which, None) is not None
            and self.params[which].get("gamma1", None) is not None
        ):
            params_config["gamma1"] = {"values": [self.params[which]["gamma1"]]}
        else:
            params_config["gamma1"] = {
                "min": GAMMA_MIN,
                "max": GAMMA_MAX,
                "scale": "log",
                "grid": GAMMA_GRID,
            }
        if (
            self.params.get(which, None) is not None
            and self.params[which].get("gamma2", None) is not None
        ):
            params_config["gamma2"] = {"values": [self.params[which]["gamma2"]]}
        else:
            params_config["gamma2"] = {
                "min": GAMMA_MIN,
                "max": GAMMA_MAX,
                "scale": "log",
                "grid": GAMMA_GRID,
            }

        # hyperparameter tuning, or use provided values
        if any([len(v.get("values", [])) != 1 for _, v in params_config.items()]):
            objectives_config = {
                "score": {"target": 0.0, "limit": 1.0, "priority": 10.0},
                "gamma1": {"target": GAMMA_MIN, "limit": GAMMA_MAX, "priority": 0.1},
                "gamma2": {"target": GAMMA_MIN, "limit": GAMMA_MAX, "priority": 0.1},
                "lambda1": {
                    "target": 0.0,
                    "limit": LAMBDA_MAX_FACTOR * len(fit_data),
                    "priority": 0.1,
                },
                "lambda2": {
                    "target": 0.0,
                    "limit": LAMBDA_MAX_FACTOR * len(fit_data),
                    "priority": 0.1,
                },
            }
            tuner = tune(
                _fit_bridge,
                params_config,
                objectives_config,
                self.num_runs,
                self.n_jobs,
            )
            params = tuner.get_best_params()
            scores = tuner.get_best_scores()
        else:
            params = self.params[which]
            scores = {
                "score": np.nan,
                "gamma1": np.nan,
                "gamma2": np.nan,
                "lambda1": np.nan,
                "lambda2": np.nan,
            }

        # fit bridge function
        bridge = method(
            params["lambda1"],
            params["lambda2"],
            params["gamma1"],
            params["gamma2"],
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
                    ("scaler", StandardScaler()),
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
                self.num_runs,
                self.n_jobs,
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
                    ("std_scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=int(degree))),
                    ("logistic", LogisticRegression(C=C, max_iter=5000)),
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
            objectives_config = {"log_loss": {"target": 0.0, "limit": 1e2}}
            tuner = tune(
                fit_prob, params_config, objectives_config, self.num_runs, self.n_jobs
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

    def fit(self, hidmed_dataset, reduce=True, split_seed=None):
        """Cross-fitting estimator."""
        data_splits = hidmed_dataset.split(self.folds + 1, seed=split_seed)

        # split data into training and validation sets
        data_splits, val_data = data_splits[:-1], data_splits[-1]

        # cross-fitting estimation
        res = []
        for i, fit_data in enumerate(data_splits):

            if self.folds > 1:
                # set up data for cross-fitting
                eval_data = data_splits[0 if i > 0 else 1].copy()
                for j, data_split in enumerate(data_splits):
                    if j == i:
                        continue
                    eval_data.extend(data_split)
            else:
                # no cross-fitting: use the same data for fitting and estimation
                eval_data = fit_data

            if self.verbose:
                print(
                    f"==== Cross-fitting fold {i+1} ({len(fit_data)}/{len(eval_data)} fit/eval)"
                )

            # estimation
            res_i = self.estimate(fit_data, eval_data, val_data)
            assert res_i.ndim == 1

            if self.verbose:
                if hasattr(res_i, "__len__"):
                    print(f"==== Estimate {i+1}: {np.mean(res_i)}, {len(res_i)} values")
                else:
                    print(f"==== Estimate {i+1}: {res_i}")

            res.append(res_i)

        res = np.hstack(res)
        reduced_estimate = np.mean(res)
        if self.verbose:
            print(
                f"==== {type(self).__name__} estimate: {reduced_estimate}, {len(res)} values"
            )

        return reduced_estimate if reduce else res

    def estimate(self, fit_data, eval_data, val_data):
        """Return point estimates for every data point."""
        raise NotImplementedError
