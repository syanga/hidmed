import numpy as np


def is_covered(psi, psi_true):
    """check if true value is covered by estimates with 95% confidence"""
    mean = np.mean(psi)
    ci = 1.96 * np.std(psi) / np.sqrt(len(psi))
    return psi_true <= mean + ci and psi_true >= mean - ci


def confidence_interval(x):
    """Calculate confidence interval"""
    return 2 * (1.96 * np.std(x) / np.sqrt(len(x)))


def calculate_mse(psi, psi_true):
    """calculate MSE of estimate"""
    bias = np.mean(psi) - psi_true
    var = np.var(psi) / len(psi)
    return var + bias**2
