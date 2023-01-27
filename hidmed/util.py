import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV


""" Create matrix out of vector """


def unvec(a):
    return np.expand_dims(a, 1) if a.ndim == 1 else a


""" Sample from a uniform distribution over two disjoint intervals """


def sample_uniform_disjoint(low, high, size):
    signs = 2 * (np.random.choice(2, size=size) - 0.5)
    vals = np.random.uniform(low=low, high=high, size=size)
    return signs * vals


""" Calcuate coverage: percentage of time ground truth value is within
    confidence interval
    X has size (num datasets, )
"""


def estimate_coverage(x, stds=2):
    mu = np.mean(x)
    stdv = np.std(x)
    return np.sum((mu - stds * stdv <= x) * (x <= mu + stds * stdv)) / len(x)


""" check if true value is covered by estimates with 95% confidence"""


def is_covered(psi, psi_true):
    mean = np.mean(psi)
    ci = 1.96 * np.std(psi) / np.sqrt(len(psi))
    return psi_true <= mean + ci and psi_true >= mean - ci


""" calculate MSE of estimate """


def calculate_mse(psi, psi_true):
    bias = np.mean(psi) - psi_true
    var = np.var(psi) / len(psi)
    return var + bias**2


def logistic_regression(y, X):
    # only 1 label present in the data
    y_vals = set(y)
    if len(y_vals) == 1:
        p = lambda y, X: y_vals.pop() * np.ones(X.shape[0])
        return p

    clf = LogisticRegression().fit(X, y)
    p = lambda y, X: clf.predict_proba(X)[:, y]
    return p


def kernel_regression(y, X, alpha=0.5, gamma=None, cv=False):
    if cv:
        reg = KernelRidge(kernel="rbf")
        param_grid = {
            "alpha": [(1 / d) * 10 ** (-a) for a in range(4) for d in range(2)],
            "gamma": [(1 / d) * 10 ** (-a) for a in range(4) for d in range(2)],
        }
        krr = HalvingGridSearchCV(reg, param_grid, cv=5).fit(X, y)
    else:
        krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma).fit(X, y)

    return lambda x: krr.predict(x)
