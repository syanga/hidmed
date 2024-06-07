import numpy as np


def confidence_interval(x):
    """Calculate confidence interval length"""
    return 2 * (1.96 * np.std(x) / np.sqrt(len(x.flatten())))


def is_covered(psi, true_psi):
    """check if true value is covered by estimates with 95% confidence"""
    mean = np.mean(psi)
    half_ci = 0.5 * confidence_interval(psi)
    return true_psi <= mean + half_ci and true_psi >= mean - half_ci


def calculate_mse(psi, true_psi):
    """calculate MSE of estimate"""
    bias = np.mean(psi) - true_psi
    var = np.var(psi) / len(psi)
    return var + bias**2


def absolute_normalized_bias(psi, true_psi):
    """calculate absolute normalized bias of estimate"""
    return np.abs(np.mean(psi) - true_psi) / np.abs(true_psi)
