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


class CrossFittingEstimatorBase:
    """Estimator that uses cross-fitting to estimate the nuisance functions"""

    def __init__(
        self,
        setup,
        folds=2,
        kernel_metric="rbf",
        num_runs=200,
        num_jobs=1,
        verbose=True,
        param_dict=None,
    ):
        self.folds = folds
        self.base_estimator_params = {
            "setup": setup,
            "kernel_metric": kernel_metric,
            "num_runs": num_runs,
            "num_jobs": num_jobs,
            "verbose": verbose,
        }
        self.verbose = verbose
        self.estimators = []

        # dictionary of parameter dictionaries, one per fold
        # if none, perform hyperparameter tuning
        if param_dict is None:
            self.param_dict = {fold: {} for fold in range(folds)}
        else:
            self.param_dict = param_dict

    def fit(self, hidmed_dataset, seed=None):
        """Fit an estimator to each fold of the data"""
        self.data_splits = hidmed_dataset.split(self.folds, seed=seed)

        # fit estimator to each fold separately
        for i in range(self.folds):
            # collect fitting and validation data for cross-fitting
            fold_data = None
            for j in range(self.folds):
                if i == j and self.folds > 1:
                    continue
                if fold_data is None:
                    fold_data = self.data_splits[j].copy()
                else:
                    fold_data.extend(self.data_splits[j].copy())
            fit_data, val_data = fold_data.split(2)

            if self.verbose:
                print(f"==== Fitting fold {i+1} ({len(fit_data)} fitting, {len(val_data)} valid.)")

            # fit estimator
            estimator = self.base_estimator(
                **self.base_estimator_params, **self.param_dict[i]
            )
            estimator.fit(fit_data, val_data)

            # store estimator
            self.estimators.append(estimator)
            self.param_dict[i] = estimator.params

        return self

    def evaluate(self, eval_data=None, reduce=True, verbose=True):
        """Estimate the quantity of interest on the provided or cached
        evaluation data"""
        res = []
        for i, estimator in enumerate(self.estimators):
            # no evaluation data manually provided
            if eval_data is None:
                # use cached fit data -- cross-fitting
                eval_data = self.data_splits[i]

            # evaluate current estimator
            res_i = estimator.evaluate(eval_data).flatten()
            res.append(res_i)

            if self.verbose and verbose:
                if hasattr(res_i, "__len__"):
                    print(f"==== Estimate {i+1}: {np.mean(res_i)}, {len(res_i)} values")
                else:
                    print(f"==== Estimate {i+1}: {res_i}")

        res = np.vstack(res)
        reduced_estimate = np.mean(res)

        if self.verbose and verbose:
            print(
                f"==== {type(self).__name__} estimate: {reduced_estimate}, {len(res.flatten())} values"
            )

        return reduced_estimate if reduce else res
