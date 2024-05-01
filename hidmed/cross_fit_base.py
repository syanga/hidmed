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
        **kwargs,
    ):
        self.folds = folds
        self.base_estimator_params = {
            "setup": setup,
            "kernel_metric": kernel_metric,
            "num_runs": num_runs,
            "num_jobs": num_jobs,
            "verbose": verbose,
            **kwargs,
        }
        self.verbose = verbose
        self.estimators = []

    def fit(self, hidmed_dataset, seed=None):
        """Fit an estimator to each fold of the data"""
        data_splits = hidmed_dataset.split(self.folds + 1, seed=seed)

        # split data into fit/eval and validation sets
        self.data_splits, self.val_data = data_splits[:-1], data_splits[-1]

        # fit estimator to each fold separately
        for i, fit_data in enumerate(self.data_splits):
            if self.verbose:
                print(f"==== Fitting fold {i+1} ({len(fit_data)} data points)")

            # fit estimator
            estimator = self.base_estimator(**self.base_estimator_params)
            estimator.fit(fit_data, self.val_data)

            # store estimator
            self.estimators.append(estimator)

        return self

    def evaluate(self, eval_data=None, reduce=True, verbose=True):
        """Estimate the quantity of interest on the provided or cached
        evaluation data"""
        res = []
        for i, estimator in enumerate(self.estimators):
            # no evaluation data manually provided
            if eval_data is None:
                if self.folds == 1:
                    # use cached fit data -- no cross-fitting
                    eval_data = self.data_splits[i]
                else:
                    # set up data for cross-fitting
                    eval_data = self.data_splits[0 if i > 0 else 1].copy()
                    for j, data_split in enumerate(self.data_splits):
                        if j == i:
                            continue
                        eval_data.extend(data_split)

            # evaluate current estimator
            res_i = estimator.evaluate(eval_data)
            assert res_i.ndim == 1, "Estimator must return a 1D array"
            res.append(res_i)

            if self.verbose and verbose:
                if hasattr(res_i, "__len__"):
                    print(f"==== Estimate {i+1}: {np.mean(res_i)}, {len(res_i)} values")
                else:
                    print(f"==== Estimate {i+1}: {res_i}")

        res = np.hstack(res)
        reduced_estimate = np.mean(res)

        if self.verbose and verbose:
            print(
                f"==== {type(self).__name__} estimate: {reduced_estimate}, {len(res)} values"
            )

        return reduced_estimate if reduce else res
