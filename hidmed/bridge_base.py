"""Estimate bridge functions in the Proximal generalized hidden mediation model
using minimax optimization and kernel methods"""

from sklearn.metrics.pairwise import pairwise_kernels


class KernelBridgeBase:
    """Estimator for the bridge function in the Proximal generalized hidden
    mediation model"""

    def __init__(self, lambda1, lambda2, gamma, treatment_prob=None):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self.treatment_prob = treatment_prob

        # fitted values
        self.alpha = None
        self.x = None

    def call_kernel(self, x, y):
        """Call the kernel function with the given data"""
        return pairwise_kernels(x, y, metric="rbf", gamma=self.gamma)

    def __call__(self, x):
        """Evaluate the bridge function at the given points"""
        return self.call_kernel(x, self.x).dot(self.alpha)

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        raise NotImplementedError

    def score(self, val_data):
        """Score the bridge function using minimax optimization"""
        raise NotImplementedError
