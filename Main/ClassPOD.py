import numpy as np
from numpy import linalg as la
from Main import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: finalise docstrings

class POD(Model):
    def __init__(self, train_array: np.array or None, val_array:  None = None, n: int = 5) -> None:
        """
        Require 2D velocity field
        :param input_: T, i, j, u -> time, i_grid, j_grid, vel u, vel v
        :param n: number of modes to include in reconstruction
        """

        # Parameters
        self.n = n  # modes to consider in reconstruction

        # POD matrices, to be generated
        self.mode_energy = None  # length n energy vector   : energy by mode
        self.phi = None      # T x i*j*u spatial mode matrix  : spatial modes
        self.a = None      # T x n temporal mode matrix : modulate with time
        self.dim = None     # dimensions

        super().__init__(train_array, val_array)

    # SKELETON FUNCTIONS: FILL (OVERWRITE) IN SUBCLASS
    def fit_model(self, train_array: np.array, val_array: np.array or None = None) -> None:  # skeleton
        """
        Fits the model on the training data: skeleton, overwrite
        :param input_: time series of inputs
        """
        self.compute(train_array)

    def get_code(self, input_: np.array) -> np.array:  # skeleton
        """
        Passes self.input through the model, returns code
        :input_: time series input
        :return: time series code
        """
        # TODO: Implement this properly; need to get new V, compatible with old mode matrix but new data
        return np.copy(self.a)

    def get_output(self, input_: np.array) -> np.array:  # skeleton
        """
        Passes self.code through the model, returns output
        :input_: time series code
        :return: time series output
        """
        self.a = input_
        return np.copy(self.reconstruct())
    # END SKELETONS

    def compute(self, input_: np.array) -> None:
        """
        Compute and store the sigma, spacial mode, and temporal mode matrices
        :return: None
        """
        dim = input_.shape  # get dimensions
        # project data; each frame is flattened into 1 vector giving T x (i*j*u) matrix
        UU = np.reshape(input_, (dim[0], dim[1] * dim[2] * dim[3]))

        m = UU.shape[0]
        C = np.matmul(np.transpose(UU), UU) / (m - 1)

        # solve eigenvalue problem
        eig, phi = la.eigh(C)

        # Sort Eigenvalues and vectors
        idx = eig.argsort()[::-1]
        eig = eig[idx]
        phi = phi[:, idx]

        # project onto modes for temporal coefficients
        a = np.matmul(UU, phi)  # contains the "code" (modal coefficients)

        # contribution of different eigenvectors -> energy
        mode_energy = eig / np.sum(eig)

        # store
        self.phi = np.copy(phi)
        self.a = np.copy(a)
        self.mode_energy = mode_energy
        self.dim = dim

    def reconstruct(self, n_modes: int or None = None) -> np.array:
        """
        Return the reconstructed flow data from the proper orthogonal decomposition, considering n most energetic modes
        :param n_modes: optional, number of nodes to consider in reconstruction
        :return: reconstructed flow data
        """
        if n_modes is None:  # overwrite if not given
            n_modes = self.n
        dim = self.dim

        print(self.phi.shape, self.a.shape, self.dim)
        # self.a = self.a[0, 0]
        # print(self.phi.shape, self.a.shape, self.dim)
        recons = np.matmul(self.a[:, :n_modes], np.transpose(self.phi[:, :n_modes]))
        recons_reshape = np.reshape(recons, (dim[0], dim[1], dim[2], dim[3]))

        return recons_reshape

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
        phi = self.phi
        dim = self.dim

        phi_spat = np.reshape(phi, (dim[1], dim[2], dim[1] * dim[2] * dim[3], dim[3]))  # contains the spatial mode
        raise NotImplemented


if __name__ == '__main__':
    u_all_2 = POD.preprocess(Nu=2, split=False)
    Model2 = POD(u_all_2, n=128)
    out2 = Model2.passthrough(Model2.input)

    slice_ = out2[0]
    fig = plt.figure()
    ax = plt.subplot(121)
    ax.contourf(slice_[:, :, 0])
    ax = plt.subplot(122)
    ax.contourf(slice_[:, :, 1])
    plt.show()
