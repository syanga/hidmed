import numpy as np


def is_covered(psi, true_psi):
    """check if true value is covered by estimates with 95% confidence"""
    mean = np.mean(psi)
    ci = 1.96 * np.std(psi) / np.sqrt(len(psi))
    return true_psi <= mean + ci and true_psi >= mean - ci


def confidence_interval(x):
    """Calculate confidence interval"""
    return 2 * (1.96 * np.std(x) / np.sqrt(len(x)))


def calculate_mse(psi, true_psi):
    """calculate MSE of estimate"""
    bias = np.mean(psi) - true_psi
    var = np.var(psi) / len(psi)
    return var + bias**2


def absolute_normalized_bias(psi, true_psi):
    """calculate absolute normalized bias of estimate"""
    return np.abs(np.mean(psi) - true_psi) / np.abs(true_psi)
