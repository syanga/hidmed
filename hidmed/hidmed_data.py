"""Dataset class for the Proximal generalized hidden mediation model"""

import numpy as np


class HidMedDataset:
    """Dataset for the Proximal generalized hidden mediation model"""

    def __init__(self, x_data, u_data, a_data, m_data, w_data, z_data, y_data):
        # dataset size should be the same for all variables
        self.n = x_data.shape[0]
        assert u_data.shape[0] == self.n
        assert a_data.shape[0] == self.n
        assert m_data.shape[0] == self.n
        assert w_data.shape[0] == self.n
        assert z_data.shape[0] == self.n
        assert y_data.shape[0] == self.n

        # dimensionality of each variable
        self.x_dim = x_data.shape[1]
        self.u_dim = u_data.shape[1]
        self.a_dim = a_data.shape[1]
        self.m_dim = m_data.shape[1]
        self.w_dim = w_data.shape[1]
        self.z_dim = z_data.shape[1]
        self.y_dim = y_data.shape[1]

        # concatenate all data
        self.data = np.hstack(
            (
                x_data,
                u_data,
                a_data,
                m_data,
                w_data,
                z_data,
                y_data,
            )
        )

        # start and end indices for each variable
        self.start_indices = {
            "x": 0,
            "u": self.x_dim,
            "a": self.x_dim + self.u_dim,
            "m": self.x_dim + self.u_dim + self.a_dim,
            "w": self.x_dim + self.u_dim + self.a_dim + self.m_dim,
            "z": self.x_dim + self.u_dim + self.a_dim + self.m_dim + self.w_dim,
            "y": self.x_dim
            + self.u_dim
            + self.a_dim
            + self.m_dim
            + self.w_dim
            + self.z_dim,
        }
        self.end_indices = {
            "x": self.x_dim,
            "u": self.x_dim + self.u_dim,
            "a": self.x_dim + self.u_dim + self.a_dim,
            "m": self.x_dim + self.u_dim + self.a_dim + self.m_dim,
            "w": self.x_dim + self.u_dim + self.a_dim + self.m_dim + self.w_dim,
            "z": self.x_dim
            + self.u_dim
            + self.a_dim
            + self.m_dim
            + self.w_dim
            + self.z_dim,
            "y": self.x_dim
            + self.u_dim
            + self.a_dim
            + self.m_dim
            + self.w_dim
            + self.z_dim
            + self.y_dim,
        }

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.n

    def __repr__(self):
        """Return a string representation of the dataset"""
        return f"HidMedDataset(n={self.n}, x_dim={self.x_dim}, u_dim={self.u_dim}, a_dim={self.a_dim}, m_dim={self.m_dim}, w_dim={self.w_dim}, z_dim={self.z_dim}, y_dim={self.y_dim})"

    def __getitem__(self, index):
        return HidMedDataset(
            self.x[index],
            self.u[index],
            self.a[index],
            self.m[index],
            self.w[index],
            self.z[index],
            self.y[index],
        )

    def copy(self):
        """Return a copy of this dataset"""
        return HidMedDataset(self.x, self.u, self.a, self.m, self.w, self.z, self.y)

    def extend(self, dataset):
        """Incorporate another dataset into this dataset"""
        assert self.x_dim == dataset.x_dim
        assert self.u_dim == dataset.u_dim
        assert self.a_dim == dataset.a_dim
        assert self.m_dim == dataset.m_dim
        assert self.w_dim == dataset.w_dim
        assert self.z_dim == dataset.z_dim
        assert self.y_dim == dataset.y_dim
        self.data = np.vstack((self.data, dataset.data))
        self.n += dataset.n

    def split(self, num_splits, seed=None):
        """Split the dataset into num_splits new datasets"""
        if seed is not None:
            np.random.seed(seed)
        inds = np.arange(self.n)
        np.random.shuffle(inds)
        split_inds = [np.sort(split) for split in np.array_split(inds, num_splits)]

        datasets = []
        for inds in split_inds:
            x_data = self.data[inds, : self.x_dim]
            u_data = self.data[inds, self.start_indices["u"] : self.end_indices["u"]]
            a_data = self.data[inds, self.start_indices["a"] : self.end_indices["a"]]
            m_data = self.data[inds, self.start_indices["m"] : self.end_indices["m"]]
            w_data = self.data[inds, self.start_indices["w"] : self.end_indices["w"]]
            z_data = self.data[inds, self.start_indices["z"] : self.end_indices["z"]]
            y_data = self.data[inds, self.start_indices["y"] : self.end_indices["y"]]
            datasets.append(
                HidMedDataset(x_data, u_data, a_data, m_data, w_data, z_data, y_data)
            )

        return datasets

    def get_data(self, variable):
        """Return data"""
        return self.data[:, self.start_indices[variable] : self.end_indices[variable]]

    @property
    def x(self):
        """Return the X data"""
        return self.get_data("x")

    @property
    def u(self):
        """Return the U data"""
        return self.get_data("u")

    @property
    def a(self):
        """Return the A data"""
        return self.get_data("a")

    @property
    def m(self):
        """Return the M data"""
        return self.get_data("m")

    @property
    def w(self):
        """Return the W data"""
        return self.get_data("w")

    @property
    def z(self):
        """Return the Z data"""
        return self.get_data("z")

    @property
    def y(self):
        """Return the Y data"""
        return self.get_data("y")
