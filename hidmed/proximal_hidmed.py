from .hidmed_dataset import *
from .rkhs import *
from .proximal_base import *
from .util import *


class ProximalHidMedRKHS(ProximalRKHSBase):
    def __init__(self, proximal_dataset, kernel_function_h, kernel_function_q):
        super().__init__(proximal_dataset, kernel_function_h, kernel_function_q)

        # precompte rkhs basis
        unvec = lambda a: np.expand_dims(a, 1) if a.ndim == 1 else a
        self.K_h = kernel_function_h
        self.K_q = kernel_function_q
        self.harg = np.hstack((unvec(self.dataset.X), unvec(self.dataset.W)))
        self.qarg = np.hstack((unvec(self.dataset.X), unvec(self.dataset.Z)))

        self.harg1 = unvec(self.dataset.W)
        self.harg2 = unvec(self.dataset.X)
        self.qarg1 = unvec(self.dataset.Z)
        self.qarg2 = unvec(self.dataset.X)

        # precompute large kernel matrices
        self.Kh = self.K_h(self.harg)
        self.Kq = self.K_q(self.qarg)

    """ estimate optimization objective parameters """

    def estimate_gs(self, which, a, idx=None):
        assert which in ["h", "q"]
        if which == "h":
            subset = self.dataset.get_A(idx) == 1
            g1 = -1.0
            g2 = self.dataset.get_Y(idx)[subset]
        else:
            where_a1 = self.dataset.get_A(idx) == 1
            where_a0 = self.dataset.get_A(idx) == 0
            xarg = unvec(self.dataset.get_X(idx))
            pax = logistic_regression(self.dataset.get_A(idx), xarg)
            p1x = pax(1, xarg)
            p0x = 1 - p1x
            g1 = where_a1 / p1x
            g2 = -1.0 * where_a0 / p0x
            subset = None
        return g1, g2, subset

    """ plugin estimator using h """

    def plugin_h(self, lambdas, alpha, gamma, n_splits, psis=(1, 2), get_raw=False):
        estimate_psi1 = psis == 1 or psis == (1, 2) or psis == [1, 2]
        estimate_psi2 = psis == 2 or psis == (1, 2) or psis == [1, 2]

        splits = self.dataset.create_crossfit_split(n_splits)
        total_samples = np.sum([len(split["eval"]) for split in splits])

        idx = 0
        psi1 = np.zeros(total_samples)
        psi2 = np.zeros(total_samples)
        for split in splits:
            # estimate h
            g1, g2, subset = self.estimate_gs("h", None, idx=split["train"])
            h = self.estimate_bridge_a(
                "h", 1, lambdas, g1=g1, g2=g2, idx=split["train"][subset]
            )

            # estimate E[h|A=0,X]
            where_a0_train = self.dataset.get_A(split["train"]) == 0
            wa0 = unvec(self.dataset.get_W(split["train"]))[where_a0_train]
            xa0 = unvec(self.dataset.get_X(split["train"]))[where_a0_train]
            ha0 = h(wa0, xa0)
            Eh = kernel_regression(ha0, xa0, alpha=alpha, gamma=gamma)

            # evaluate functions
            xeval = unvec(self.dataset.get_X(split["eval"]))

            if estimate_psi2:
                # estimate p(A=1|x)
                pax = logistic_regression(
                    self.dataset.get_A(split["train"]),
                    unvec(self.dataset.get_X(split["train"])),
                )
                p1x = pax(1, xeval)
                arg = Eh(xeval) * p1x
                psi2[idx : idx + len(split["eval"])] = arg

            if estimate_psi1:
                arg = Eh(xeval)
                psi1[idx : idx + len(split["eval"])] = arg

            idx += len(split["eval"])

        if estimate_psi1 and estimate_psi2:
            if get_raw:
                return psi1, psi2

            return np.mean(psi1), np.mean(psi2)
        elif estimate_psi1:
            if get_raw:
                return psi1

            return np.mean(psi1)

        if get_raw:
            return psi2

        return np.mean(psi2)

    """ plugin estimator using q """

    def plugin_q(self, lambdas, alpha, gamma, n_splits, psis=(1, 2), get_raw=False):
        estimate_psi1 = psis == 1 or psis == (1, 2) or psis == [1, 2]
        estimate_psi2 = psis == 2 or psis == (1, 2) or psis == [1, 2]

        splits = self.dataset.create_crossfit_split(n_splits)
        total_samples = np.sum([len(split["eval"]) for split in splits])

        idx = 0
        psi1 = np.zeros(total_samples)
        psi2 = np.zeros(total_samples)
        for split in splits:
            # estimate q
            g1, g2, _ = self.estimate_gs("q", None, idx=split["train"])
            q = self.estimate_bridge_a(
                "q", 1, lambdas, g1=g1, g2=g2, idx=split["train"]
            )

            # estimate E[yq|A=1,X]
            a1 = np.squeeze(self.dataset.get_A(split["train"]) == 1)
            y = self.dataset.get_Y(split["train"])
            z = self.dataset.get_Z(split["train"])
            x = self.dataset.get_X(split["train"])
            Eyq = kernel_regression(
                np.squeeze(y[a1] * q(z[a1], x[a1])),
                unvec(x[a1]),
                alpha=alpha,
                gamma=gamma,
            )

            # # estimate p(a'|X)
            # pax = logistic_regression(np.squeeze(self.dataset.get_A(split["train"])), unvec(self.dataset.get_X(split["train"])))

            xeval = unvec(self.dataset.get_X(split["eval"]))
            # p1x = pax(1, xeval)

            if estimate_psi1:
                # psi1 += np.mean(where_1 * eval_y * eval_q / p1x) / len(splits)
                arg = Eyq(xeval)
                psi1[idx : idx + len(split["eval"])] = arg
                # psi1 += np.mean(Eyq(xeval)) / len(splits)

            if estimate_psi2:
                where_1 = np.squeeze(self.dataset.get_A(split["eval"]) == 1)
                eval_y = np.squeeze(self.dataset.get_Y(split["eval"]))
                zeval = unvec(self.dataset.get_Z(split["eval"]))
                eval_q = q(zeval, xeval)

                arg = where_1 * eval_y * eval_q
                psi2[idx : idx + len(split["eval"])] = arg
                # psi2 += np.mean(where_1 * eval_y * eval_q) / len(splits)

            idx += len(split["eval"])

        if estimate_psi1 and estimate_psi2:
            if get_raw:
                return psi1, psi2

            return np.mean(psi1), np.mean(psi2)
        elif estimate_psi1:
            if get_raw:
                return psi1

            return np.mean(psi1)

        if get_raw:
            return psi2

        return np.mean(psi2)

    """ plugin estimator using q with propensity scores """

    def plugin_q_prop(self, lambdas, alpha, n_splits, psis=(1, 2), get_raw=False):
        estimate_psi1 = psis == 1 or psis == (1, 2) or psis == [1, 2]
        estimate_psi2 = psis == 2 or psis == (1, 2) or psis == [1, 2]

        splits = self.dataset.create_crossfit_split(n_splits)
        total_samples = np.sum([len(split["eval"]) for split in splits])

        idx = 0
        psi1 = np.zeros(total_samples)
        psi2 = np.zeros(total_samples)
        for split in splits:
            # estimate q
            g1, g2, _ = self.estimate_gs("q", None, idx=split["train"])
            q = self.estimate_bridge_a(
                "q", 1, lambdas, g1=g1, g2=g2, idx=split["train"]
            )

            # estimate p(a'|X)
            pax = logistic_regression(
                np.squeeze(self.dataset.get_A(split["train"])),
                unvec(self.dataset.get_X(split["train"])),
            )

            xeval = unvec(self.dataset.get_X(split["eval"]))
            p1x = pax(1, xeval)

            where_1 = np.squeeze(self.dataset.get_A(split["eval"]) == 1)
            eval_y = np.squeeze(self.dataset.get_Y(split["eval"]))
            zeval = unvec(self.dataset.get_Z(split["eval"]))
            eval_q = q(zeval, xeval)

            if estimate_psi1:
                # psi1 += np.mean(where_1 * eval_y * eval_q / p1x) / len(splits)
                psi1[idx : idx + len(split["eval"])] = where_1 * eval_y * eval_q / p1x
            if estimate_psi2:
                # psi2 += np.mean(where_1 * eval_y * eval_q) / len(splits)
                psi2[idx : idx + len(split["eval"])] = where_1 * eval_y * eval_q

        if estimate_psi1 and estimate_psi2:
            if get_raw:
                return psi1, psi2

            return np.mean(psi1), np.mean(psi2)
        elif estimate_psi1:
            if get_raw:
                return psi1

            return np.mean(psi1)

        if get_raw:
            return psi2

        return np.mean(psi2)

    """ plugin estimator using q """

    def double_robust(
        self, lambdas_h, lambdas_q, alpha, gamma, n_splits, psis=(1, 2), get_raw=False
    ):
        estimate_psi1 = psis == 1 or psis == (1, 2) or psis == [1, 2]
        estimate_psi2 = psis == 2 or psis == (1, 2) or psis == [1, 2]

        splits = self.dataset.create_crossfit_split(n_splits)
        total_samples = np.sum([len(split["eval"]) for split in splits])

        idx = 0
        psi1 = np.zeros(total_samples)
        psi2 = np.zeros(total_samples)
        for split in splits:
            # estimate h
            g1, g2, subset = self.estimate_gs("h", None, idx=split["train"])
            h = self.estimate_bridge_a(
                "h", 1, lambdas_h, g1=g1, g2=g2, idx=split["train"][subset]
            )

            # estimate q
            g1, g2, _ = self.estimate_gs("q", None, idx=split["train"])
            q = self.estimate_bridge_a(
                "q", 1, lambdas_q, g1=g1, g2=g2, idx=split["train"]
            )

            # estimate eta
            where_a0 = self.dataset.get_A(split["train"]) == 0
            wtrain0 = unvec(self.dataset.get_W(split["train"]))[where_a0]
            xtrain0 = unvec(self.dataset.get_X(split["train"]))[where_a0]
            eta = kernel_regression(
                h(wtrain0, xtrain0), xtrain0, alpha=alpha, gamma=gamma
            )

            # estimate p(A|X)
            pax = logistic_regression(
                self.dataset.get_A(split["train"]),
                unvec(self.dataset.get_X(split["train"])),
            )

            # evaluate estimators
            A = self.dataset.get_A(split["eval"])
            Y = self.dataset.get_Y(split["eval"])
            W = unvec(self.dataset.get_W(split["eval"]))
            Z = unvec(self.dataset.get_Z(split["eval"]))
            X = unvec(self.dataset.get_X(split["eval"]))

            p1x = pax(1, X)
            p0x = pax(0, X)
            qzx = q(Z, X)
            hwx = h(W, X)
            etax = eta(X)

            if estimate_psi1:
                # psi1 += np.mean((A / p1x) * qzx * (Y - hwx) + ((1 - A)/p0x) * (hwx - etax) + etax) / len(splits)
                # arg = (A / p1x) * qzx * (Y - hwx) + ((1 - A)/p0x) * (hwx - etax) + etax
                psi1[idx : idx + len(split["eval"])] = (
                    (A / p1x) * qzx * (Y - hwx) + ((1 - A) / p0x) * (hwx - etax) + etax
                )

            if estimate_psi2:
                # psi2 += np.mean(A * qzx * (Y - hwx) + (1-A)*(p1x / p0x) * (hwx - etax) + A*etax) / len(splits)
                # arg = A * qzx * (Y - hwx) + (1-A)*(p1x / p0x) * (hwx - etax) + A*etax
                psi2[idx : idx + len(split["eval"])] = (
                    A * qzx * (Y - hwx)
                    + (1 - A) * (p1x / p0x) * (hwx - etax)
                    + A * etax
                )

            idx += len(split["eval"])

        if estimate_psi1 and estimate_psi2:
            if get_raw:
                return psi1, psi2

            return np.mean(psi1), np.mean(psi2)
        elif estimate_psi1:
            if get_raw:
                return psi1

            return np.mean(psi1)

        if get_raw:
            return psi2

        return np.mean(psi2)
