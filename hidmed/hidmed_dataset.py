import numpy as np


class HidMedDataset:
    def __init__(self, X, U, A, M, W, Z, Y):
        self.X = X
        self.U = U
        self.A = A
        self.M = M
        self.W = W
        self.Z = Z
        self.Y = Y
        self.n = X.shape[0]
        assert U.shape[0] == self.n
        assert A.shape[0] == self.n
        assert M.shape[0] == self.n
        assert W.shape[0] == self.n
        assert Z.shape[0] == self.n
        assert Y.shape[0] == self.n

    """ split the dataset into n_split groups for cross-fitting """

    def create_crossfit_split(self, n_splits, cache=True):
        if n_splits < 0:
            idx = np.arange(self.n)
            split_indices = [{"train": idx, "eval": idx}]
        else:
            idx = np.arange(self.n)
            np.random.shuffle(idx)
            split_size = int(np.floor(self.n / n_splits))
            split_indices = []
            for i in range(n_splits):
                start = i * split_size
                end = (i + 1) * split_size
                if i == n_splits - 1 and end < self.n - 1:
                    end = self.n - 1
                eval_idx = np.sort(idx[start:end])
                train_idx = np.sort(np.hstack((idx[: max(0, start)], idx[end:])))
                assert len(eval_idx) + len(train_idx) == self.n
                split_indices.append({"train": train_idx, "eval": eval_idx})

        if cache:
            self.split_indices = split_indices

        return split_indices

    def get_X(self, idx=None):
        if idx is None:
            return self.X
        return self.X[idx]

    def get_U(self, idx=None):
        if idx is None:
            return self.U
        return self.U[idx]

    def get_A(self, idx=None):
        if idx is None:
            return self.A
        return self.A[idx]

    def get_M(self, idx=None):
        if idx is None:
            return self.M
        return self.M[idx]

    def get_W(self, idx=None):
        if idx is None:
            return self.W
        return self.W[idx]

    def get_Z(self, idx=None):
        if idx is None:
            return self.Z
        return self.Z[idx]

    def get_Y(self, idx=None):
        if idx is None:
            return self.Y
        return self.Y[idx]
