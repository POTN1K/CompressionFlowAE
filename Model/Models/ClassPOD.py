import numpy as np
from numpy import linalg as LA
from Model.ParentClass import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class POD(Model):
    def __init__(self, u_all: np.array, n: int = 100, hot_start: int = False) -> None:
        """
        Require 2D velocity field
        :param u_all: T, i, j, u, v -> time, i_grid, j_grid, vel u, vel v
        :param n: max number of modes to generate
        :param hot_start: bool -> immediately run self.compute
        """
        # Parameters
        self.n = n  # max number of modes to consider in generation

        # Dimensions
        self.dim_u = len(np.shape(u_all)) - 3
        if self.dim_u == 1:
            self.dim_T, self.dim_x, self.dim_y, self.dim_u = np.shape(u_all)
            self.dim_v = None
        else:
            self.dim_T, self.dim_x, self.dim_y, self.dim_u, self.dim_v = np.shape(u_all)
        self.dim_M = self.dim_x*self.dim_y

        if self.n > self.dim_M:  # check validity of n
            raise Exception('Max number of modes n out of range')

        # Data
        self.input_data = u_all

        # POD matrices, to be generated
        self.mode_energy = None  # length n energy vector   : energy by mode
        self.U = None      # M x n x u spatial mode matrix  : spatial modes
        self.V = None      # n x T x u temporal mode matrix : modulate with time

        # decoding: _decoded can be generated from variable n modes
        self._decoded = None
        self._n_decoded = None

        # hot start: generate POD matrices
        if hot_start:
            self.compute()

    def compute(self) -> None:
        """
        Compute and store the sigma, spacial mode, and temporal mode matrices
        :return: None
        """
        dim = self.input_data.shape
        UU = np.reshape(self.input_data[:dim[0], :], (dim[0], dim[1] * dim[2] * dim[3]))
        m = UU.shape[0]
        C = np.matmul(np.transpose(UU), UU) / (m - 1)

        # solve eigenvalue problem
        eig, phi = LA.eigh(C)
        print(eig.shape)

        # Sort Eigenvalues and vectors
        idx = eig.argsort()[::-1]
        eig = eig[idx]
        phi = phi[:, idx]

        # project onto modes for temporal coefficients
        self.V = np.matmul(UU, phi)  # contains the "code" (modal coefficients)

        self.U = np.reshape(phi, (dim[1], dim[2], dim[1] * dim[2]))  # contains the spatial mode

        # contribution of different eigenvectors -> energy
        self.mode_energy = eig / np.sum(eig)

    def reconstruct(self) -> np.array:
        """
        Return the reconstructed flow data from the proper orthogonal decomposition
        :return: reconstructed flow data
        """
        return np.matmul(self.V[:, self._n_decoded], np.transpose(self.U[:, self._n_decoded]))

    @property
    def encoded(self):
        return self.V  # modal coefficients modulate the modes in time: this is the encoded data

    @property
    def decoded(self, n: int = None) -> np.array:
        """
        Returns the reconstructed flow data using the n most energetic modes
        :param n: int, number of modes to use in reconstruction
        :return: reconstructed flow data
        """
        if n is None:   # set n for decoding
            if self._n_decoded is not None:
                n = self._n_decoded
            else:
                n = self.n

        if self._decoded is None:  # first time decoding
            self._n_decoded = n
            self._decoded = self.reconstruct()

        elif n != self._n_decoded:  # previously decoded for other n -> overwrite
            self._n_decoded = n
            self._decoded = self.reconstruct()

        # _n_decoded is current n and _decoded is generated

        return self._decoded

    def plot_contributions(self) -> None:
        """
        Plot the energy of each mode
        """
        plt.figure()
        plt.semilogy(np.diag(self.mode_energy))
        plt.show()

    def plot_mode(self, n: int) -> None:
        """
        :param n: int, node ID to plot
        Plot the components of a specific spatial modes
        """
        raise NotImplemented

    def save_to_file(self) -> None:
        """
        Stretch goal, maybe implement later. Would also need to implement loading properly.
        """
        raise NotImplemented


if __name__ == '__main__':
    u_all = np.concatenate(POD.preprocess(nu=1))
    Model = POD(u_all, hot_start=True)
    Model.plot_contributions()


