from .rkhs import *
from .util import *
from scipy.optimize import minimize, Bounds


class ProximalRKHSBase:
    def __init__(self, proximal_dataset, kernel_function_h, kernel_function_q):
        self.dataset = proximal_dataset

        # precompte rkhs basis
        self.K_h = kernel_function_h
        self.K_q = kernel_function_q

        self.harg = np.hstack((unvec(self.dataset.X), unvec(self.dataset.M)))
        self.qarg = np.hstack((unvec(self.dataset.X), unvec(self.dataset.Z)))

        self.harg1 = unvec(self.dataset.M)
        self.harg2 = unvec(self.dataset.X)
        self.qarg1 = unvec(self.dataset.Z)
        self.qarg2 = unvec(self.dataset.X)

        # precompute large kernel matrices
        self.Kh = self.K_h(self.harg)
        self.Kq = self.K_q(self.qarg)

    """ get subsampled kernel matrices """

    def get_Kh(self, idx=None):
        if idx is None:
            return self.Kh
        else:
            return self.Kh[idx, :][:, idx]

    def get_Kq(self, idx=None):
        if idx is None:
            return self.Kq
        else:
            return self.Kq[idx, :][:, idx]

    """ estimate bridge function """

    def estimate_bridge(
        self,
        which,
        lambdas1=None,
        lambdas0=None,
        idx=None,
        g11=None,
        g21=None,
        g10=None,
        g20=None,
    ):
        assert which in ["h", "q"]
        f1 = self.estimate_bridge_a(which, 1, lambdas1, g1=g11, g2=g21, idx=idx)
        f0 = self.estimate_bridge_a(which, 0, lambdas0, g1=g10, g2=g20, idx=idx)
        return lambda w, a, x: f1(w, x) if a == 1 else f0(w, x)

    """ estimate bridge function with a fixed and hyperparameter tuning """

    def optimize_lambdas(self, which, a, xfit_splits, maxfev=10, grid_search=False):
        assert which in ["h", "q"]
        assert a == 1 or a == 0

        # test and validation splits
        splits = self.dataset.create_crossfit_split(xfit_splits)
        assert len(splits) >= 2
        idx_train = splits[0]["train"]

        # calculate average performance of lambda combination
        def obj(l):
            metric = 0
            for i in range(xfit_splits):
                idx_train = splits[i]["train"]
                idx_valid = splits[
                    np.random.choice([k for k in range(xfit_splits) if k != i])
                ]["train"]

                # fit on training
                g1, g2, subset = self.estimate_gs(which, a, idx=idx_train)
                idx_train_selected = (
                    idx_train[subset] if subset is not None else idx_train
                )
                _, alpha = self.estimate_bridge_a(
                    which,
                    a,
                    l,
                    g1=g1,
                    g2=g2,
                    idx=idx_train_selected,
                    get_f=False,
                    get_params=True,
                )

                # evaluate
                g1_cv, g2_cv, subset = self.estimate_gs(which, a, idx=idx_valid)
                idx_valid_selected = (
                    idx_valid[subset] if subset is not None else idx_valid
                )

                Kh_val = (self.get_Kh if which == "h" else self.get_Kq)(
                    idx=idx_train_selected
                )
                Kf_val = (self.get_Kq if which == "h" else self.get_Kh)(
                    idx=idx_valid_selected
                )

                h_val = alpha @ (self.K_h if which == "h" else self.K_q)(
                    (self.harg if which == "h" else self.qarg)[idx_train_selected],
                    (self.harg if which == "h" else self.qarg)[idx_valid_selected],
                )

                beta = solve_maxf(Kf_val, g1_cv, g2_cv, h_val, l[1])
                f_val = beta @ Kf_val

                metric += (
                    np.mean(f_val * (g1_cv + h_val + g2_cv) - f_val**2)
                    - l[1] * beta.T @ Kf_val @ beta
                    + l[0] * alpha.T @ Kh_val @ alpha
                )

            return metric / xfit_splits

        # initial guess
        lambdas0 = 1e-6 * np.ones(2)

        # do basic grid search
        if grid_search:
            best_val = np.inf
            best_lambda = lambdas0
            for lambda_q in np.geomspace(1e-5, 1e2, int(np.sqrt(maxfev))):
                for lambda_h in np.geomspace(1e-5, 1e2, int(np.sqrt(maxfev))):
                    lambdas = np.array([lambda_h, lambda_q]) / (len(idx_train) ** 0.8)
                    val = obj(lambdas)
                    if val < best_val:
                        best_val = val
                        best_lambda = lambdas
            # print(best_val)
            return best_lambda

        # use derivative free optimization
        res = minimize(
            obj,
            lambdas0,
            method="nelder-mead",
            options={
                "maxfev": maxfev,
            },
            bounds=Bounds([1e-12, 1e-12], [1e1, 1e1]),
        )
        lambda_opt = res.x

        return lambda_opt

    """ estimate bridge function with a fixed """

    def estimate_bridge_a(
        self,
        which,
        a,
        lambdas,
        g1=None,
        g2=None,
        idx=None,
        get_f=False,
        get_params=False,
    ):
        assert which in ["h", "q"]

        if which == "h":
            karg = self.harg if idx is None else self.harg[idx]
            kargf = self.qarg if idx is None else self.qarg[idx]
        else:
            karg = self.qarg if idx is None else self.qarg[idx]
            kargf = self.harg if idx is None else self.harg[idx]

        if get_f:
            alpha, beta = self.estimate_params(
                which, a, lambdas, g1=g1, g2=g2, idx=idx, get_beta=True
            )
            h = lambda w, x: alpha @ (self.K_h if which == "h" else self.K_q)(
                karg, np.hstack((unvec(x), unvec(w)))
            )
            f = lambda z, x: beta @ (self.K_h if which == "h" else self.K_q)(
                kargf, np.hstack((unvec(x), unvec(z)))
            )
            return (h, f, alpha, beta) if get_params else (h, f)

        alpha = self.estimate_params(which, a, lambdas, g1=g1, g2=g2, idx=idx)
        h = lambda w, x: alpha @ (self.K_h if which == "h" else self.K_q)(
            karg, np.hstack((unvec(x), unvec(w)))
        )
        return (h, alpha) if get_params else h

    """ estimate bridge function parameters """

    def estimate_params(
        self, which, a, lambdas, g1=None, g2=None, idx=None, get_beta=False
    ):
        assert a == 1 or a == 0
        assert which in ["h", "q"]
        if g1 is None or g2 is None:
            g1, g2, subset = self.estimate_gs(which, a, idx)
            if idx is None:
                # index on subset
                idx = subset
            elif subset is not None:
                # index on intersection of idx and subset
                idx *= subset

        K1 = self.get_Kh(idx) if which == "h" else self.get_Kq(idx)
        K2 = self.get_Kq(idx) if which == "h" else self.get_Kh(idx)
        alpha, beta = solve_kkt(K1, K2, g1, g2, lambdas[0], lambdas[1])
        if get_beta:
            return alpha, beta
        return alpha

    """ estimate optimization objective parameters """

    def estimate_gs(self, which, a, idx=None, param_depot={}):
        raise NotImplementedError
