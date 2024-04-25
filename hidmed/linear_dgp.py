import numpy as np
from scipy.special import expit
from .hidmed_data import HidMedDataset


def sample_uniform_disjoint(low, high, size):
    """Sample uniformly from [-high, -low] \cup [low, high]"""
    signs = 2 * (np.random.choice(2, size=size) - 0.5)
    vals = np.random.uniform(low=low, high=high, size=size)
    return signs * vals


class LinearHidMedDGP:
    """Linear data generating process for the Proximal generalized hidden mediation
    model"""

    def __init__(
        self,
        xdim,
        zdim,
        wdim,
        mdim,
        udim,
        ydim=1,
        l=1.0,
        u=2.0,
        var=0.1,
        setup="c",
        nonnegative=False,
        azwy_nonnegative=False,
        seed=0,
    ):
        np.random.seed(seed)

        self.xdim = xdim
        self.zdim = zdim
        self.wdim = wdim
        self.mdim = mdim
        self.udim = udim
        self.ydim = ydim

        sampler = np.random.uniform if nonnegative else sample_uniform_disjoint
        azwy_sampler = np.random.uniform if azwy_nonnegative else sampler

        # p(A=1|X,U) = 1/(1 + exp(X @ Wxa + U @ Wua))
        self.Wxa = sampler(low=0.4 * l, high=0.4 * u, size=(xdim, 1))
        self.Wua = sampler(low=0.4 * l, high=0.4 * u, size=(udim, 1))
        # self.Wxa = sampler(low=l, high=u, size=(xdim, 1))
        # self.Wua = sampler(low=l, high=u, size=(udim, 1))

        # M = X @ Wxm + A @ Wam + epsm
        self.Wxm = sampler(low=l, high=u, size=(xdim, mdim))
        self.Wam = sampler(low=l, high=u, size=(1, mdim))

        # Y = X @ Wxy + U @ Wuy + A @ Way + M @ Wmy + W @ Wwy + epsy
        self.Wxy = sampler(low=l, high=u, size=(xdim, ydim))
        self.Wuy = sampler(low=l, high=u, size=(udim, ydim))
        self.Wmy = sampler(low=l, high=u, size=(mdim, ydim))
        self.Way = sampler(low=l, high=u, size=(1, ydim))
        # self.Way = 2 * np.ones((1, ydim))
        self.Wwy = azwy_sampler(low=l, high=u, size=(wdim, ydim))

        # Z = M @ Wmz + X @ Wxz + A @ Waz + epsz
        self.Wmz = sampler(low=l, high=u, size=(mdim, zdim))
        self.Wxz = sampler(low=l, high=u, size=(xdim, zdim))
        self.Waz = azwy_sampler(low=l, high=u, size=(1, zdim))

        # W = M @ Wmw + X @ Wxw + epsw
        self.Wmw = sampler(low=l, high=u, size=(mdim, wdim))
        self.Wxw = sampler(low=l, high=u, size=(xdim, wdim))

        # three models
        assert setup in ["a", "b", "c"]
        self.setup = setup
        if setup == "a":
            # Proximal hidden mediation model
            self.Wua *= 0
            self.Wuy *= 0
        elif setup == "b":
            # Proximal hidden front-door model
            self.Way *= 0
        # otherwise, Proximal generalized hidden mediation model

        # covariances
        self.xcov = np.eye(xdim)
        self.ucov = np.eye(udim)

        self.mcov = var * np.eye(mdim)
        self.ycov = var * np.eye(ydim)
        self.zcov = var * np.eye(zdim)
        self.wcov = var * np.eye(wdim)

    def sample_dataset(self, n, seed=None):
        """Sample a dataset of size n"""
        np.random.seed(0 if seed is None else seed)
        epsm = np.random.multivariate_normal(np.zeros(self.mdim), self.mcov, n)
        epsy = np.random.multivariate_normal(np.zeros(self.ydim), self.ycov, n)
        epsz = np.random.multivariate_normal(np.zeros(self.zdim), self.zcov, n)
        epsw = np.random.multivariate_normal(np.zeros(self.wdim), self.wcov, n)

        X = np.random.multivariate_normal(np.zeros(self.xdim), self.xcov, n)
        U = np.random.multivariate_normal(np.zeros(self.udim), self.ucov, n)

        A = np.random.binomial(1, expit(X @ self.Wxa + U @ self.Wua))
        M = X @ self.Wxm + A * self.Wam + epsm
        W = M @ self.Wmw + X @ self.Wxw + epsw
        Y = (
            X @ self.Wxy
            + U @ self.Wuy
            + A @ self.Way
            + M @ self.Wmy
            + W @ self.Wwy
            + epsy
        )
        Z = M @ self.Wmz + X @ self.Wxz + A @ self.Waz + epsz

        return HidMedDataset(X, U, A, M, W, Z, Y)

    def true_psi(self, n=100_000):
        """Compute true value of \psi^{a',a} for the given setup"""
        if self.setup == "a":
            return self.Way.item()
        else:
            data = self.sample_dataset(n)
            p1 = np.mean(data.a)
            Wx = (
                self.Wxy
                + self.Wxw @ self.Wwy
                + self.Wxm @ (self.Wmy + self.Wmw @ self.Wwy)
            )
            psi2 = p1 * np.mean(self.Way + data.x[(data.a == 1)[:, 0]] @ Wx).item()
            psi2 += (
                np.mean(
                    data.u / (1 + np.exp(data.x @ self.Wxa + data.u @ self.Wua)), axis=0
                )
                @ self.Wuy
            )
            return psi2

    def diagnostics(self, n=100_000):
        """Check that p(A=1|X,U) is bounded away from 0 and 1"""
        data = self.sample_dataset(n)
        probs = expit(data.x @ self.Wxa + data.u @ self.Wua)
        return np.min(probs), np.max(probs)
