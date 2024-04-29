"""Estimate bridge functions in the Proximal generalized hidden mediation model
using minimax optimization and kernel methods"""

from sklearn.metrics.pairwise import pairwise_kernels


class KernelBridgeBase:
    """Estimator for the bridge function in the Proximal generalized hidden
    mediation model"""

    def __init__(self, lambda1, lambda2, gamma1, gamma2, treatment_prob=None):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.treatment_prob = treatment_prob

        # fitted values
        self.alpha = None
        self.x = None
        self.beta = None
        self.xf = None

    def kernel1(self, x, y):
        """Call the kernel function with the given data"""
        return pairwise_kernels(x, y, metric="rbf", gamma=self.gamma1)

    def kernel2(self, x, y):
        """Call the kernel function with the given data"""
        return pairwise_kernels(x, y, metric="rbf", gamma=self.gamma2)

    def __call__(self, x):
        """Evaluate the bridge function at the given points"""
        return self.kernel1(x, self.x).dot(self.alpha)

    def f(self, xf):
        """Evaluate the bridge function at the given points"""
        return self.kernel2(xf, self.xf).dot(self.beta)

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        raise NotImplementedError

    def score(self, val_data):
        """Score the bridge function using minimax optimization"""
        raise NotImplementedError
