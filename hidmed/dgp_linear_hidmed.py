import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal
from .hidmed_dataset import HidMedDataset
from .util import *
from tqdm import tqdm


class LinearHidMedDataset:
    def __init__(
        self,
        xdim,
        zdim,
        wdim,
        mdim,
        udim,
        ydim=1,
        l=0.5,
        u=1.0,
        setup="c",
        nonnegative=True,
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

        # weights
        self.Wxa = sampler(low=l, high=u, size=(xdim, 1))
        self.Wua = sampler(low=l, high=u, size=(udim, 1, 1))

        self.Wxm = sampler(low=l, high=u, size=(xdim, mdim))
        self.Wam = sampler(low=l, high=u, size=(1, mdim))

        self.Wxy = sampler(low=l, high=u, size=(xdim, ydim))
        self.Wuy = sampler(low=l, high=u, size=(udim, ydim))
        self.Wmy = sampler(low=l, high=u, size=(mdim, ydim))
        self.Way = 2 * np.ones((1, ydim))
        self.Wwy = azwy_sampler(low=l, high=u, size=(wdim, ydim))

        self.Wmz = sampler(low=l, high=u, size=(mdim, zdim))
        self.Wxz = sampler(low=l, high=u, size=(xdim, zdim))
        self.Waz = azwy_sampler(low=l, high=u, size=(1, zdim))

        self.Wmw = sampler(low=l, high=u, size=(mdim, wdim))
        self.Wxw = sampler(low=l, high=u, size=(xdim, wdim))

        assert setup in ["a", "b", "c"]
        self.setup = setup
        if setup == "a":
            self.Wua *= 0
            self.Wuy *= 0
        elif setup == "b":
            self.Way *= 0

        # covariances
        self.xcov = np.eye(xdim)
        self.ucov = np.eye(udim)

        self.mcov = 0.1 * np.eye(mdim)
        self.ycov = 0.1 * np.eye(ydim)
        self.zcov = 0.1 * np.eye(zdim)
        self.wcov = 0.1 * np.eye(wdim)

    def sample_dataset(self, n, seed=None):
        np.random.seed(seed)
        epsm = np.random.multivariate_normal(np.zeros(self.mdim), self.mcov, n)
        epsy = np.random.multivariate_normal(np.zeros(self.ydim), self.ycov, n)
        epsz = np.random.multivariate_normal(np.zeros(self.zdim), self.zcov, n)
        epsw = np.random.multivariate_normal(np.zeros(self.wdim), self.wcov, n)

        X = np.random.multivariate_normal(np.zeros(self.xdim), self.xcov, n)
        U = np.random.multivariate_normal(np.zeros(self.udim), self.ucov, n)
        A = np.random.binomial(1, expit(X @ self.Wxa + U @ self.Wua))
        M = X @ self.Wxm + A * self.Wam + epsm
        W = M @ self.Wmw + X @ self.Wxw + epsw
        Z = M @ self.Wmz + X @ self.Wxz + A @ self.Waz + epsz
        Y = (
            X @ self.Wxy
            + A @ self.Way
            + M @ self.Wmy
            + W @ self.Wwy
            + U @ self.Wuy
            + epsy
        )

        return HidMedDataset(
            X.squeeze(),
            U.squeeze(),
            A.squeeze(),
            M.squeeze(),
            W.squeeze(),
            Z.squeeze(),
            Y.squeeze(),
        )

    def true_psi(self, n=10000):
        if self.setup == "a":
            return self.psi1()
        else:
            return self.psi2(n)

    def psi1(self):
        assert self.setup == "a"
        return self.Way[0, 0]

    def psi2(self, n=10000):
        assert self.setup in ["b", "c"]
        data = self.sample_dataset(n)
        return (
            self.Way * np.mean(np.mean(data.get_A()))
            - np.mean((data.get_A() == 0) * data.get_Y())
        ).item()

    def psi(self, samples=10000, alpha=0.5):
        data = self.sample_dataset(samples)
        eval_data = self.sample_dataset(samples)

        # estimate E[Y|a',m,x]
        where_1 = np.squeeze(data.get_A() == 1)
        regressor = np.hstack(
            (unvec(data.get_M(idx=where_1)), unvec(data.get_X(idx=where_1)))
        )
        krr = kernel_regression(data.get_Y(idx=where_1), regressor, alpha=alpha)

        # estimate p(a'|x)
        p1 = logistic_regression(np.squeeze(data.get_A()), unvec(data.get_X()))

        # # p(m|a,x)
        # print(np.squeeze(eval_data.get_X(idx=1).reshape((-1,1)) @ self.Wxm).shape)
        if self.xdim > 1:
            M = np.array(
                [
                    np.random.multivariate_normal(
                        eval_data.get_X(idx=i) @ self.Wxm, self.mcov
                    )
                    for i in range(samples)
                ]
            )
        else:
            M = np.array(
                [
                    np.random.multivariate_normal(
                        np.dot(self.Wxm, eval_data.get_X(idx=i)).reshape((1,)),
                        self.mcov,
                    )
                    for i in range(samples)
                ]
            )

        Ey1mx = krr(np.hstack((unvec(M), unvec(eval_data.get_X()))))

        # pmax_vals = np.array([pmax(i) for i in range(samples)])
        psi1 = np.mean(Ey1mx)
        psi2 = np.mean(Ey1mx * p1(1, unvec(eval_data.get_X())))

        return psi1, psi2

    def psih(self, samples=2000):
        Ww = (self.Wmy + self.Wwy @ self.Wmw) @ np.linalg.inv(self.Wmw)
        Wx = self.Wxy + self.Wwy @ self.Wxw - Ww @ self.Wxw

        # E[U|m, a', x] = E[U|a',x]
        p1xu = lambda x, u: 1 / (1 + np.exp(np.dot(x, self.Wxa) + np.dot(u, self.Wua)))

        def EU(x):
            U = np.random.multivariate_normal(
                np.zeros(self.udim), self.ucov, size=samples
            )
            U2 = np.random.multivariate_normal(
                np.zeros(self.udim), self.ucov, size=samples
            )
            return np.mean([u * p1xu(x, u) for u in U]) / np.mean(
                [p1xu(x, u) for u in U2]
            )

        # p(A=1|X) regression
        data = self.sample_dataset(samples)
        p1 = logistic_regression(np.squeeze(data.get_A()), unvec(data.get_X()))

        # h function
        def h(w, x):
            return (
                np.dot(Ww, w)
                + np.dot(Wx, x)
                + self.Way
                + (0 if self.setup == "a" else EU(x))
            )

        psi1_est = 0.0
        psi2_est = 0.0
        for _ in tqdm(range(samples)):
            x = np.random.multivariate_normal(np.zeros(self.xdim), self.xcov)

            what = np.dot(self.Wxw + self.Wmw @ self.Wxm, x)
            whatcov = self.wcov + self.Wmw @ self.mcov @ (self.Wmw).T
            w = np.random.multivariate_normal(what, whatcov)

            psi1_est += h(w, x) / samples
            psi2_est += h(w, x) * p1(1, unvec(x)) / samples

        return psi1_est.item(), psi2_est.item(), h
