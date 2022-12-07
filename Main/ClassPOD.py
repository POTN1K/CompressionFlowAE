import numpy as np
from numpy import linalg as LA
from Main import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class POD(Model):
    def __init__(self, input_: np.array or None, n: int = 5) -> None:
        """
        Require 2D velocity field
        :param input_: T, i, j, u -> time, i_grid, j_grid, vel u, vel v
        :param n: number of modes to include in reconstruction
        """

        # Parameters
        self.n = n  # modes to consider in reconstruction

        # POD matrices, to be generated
        self.mode_energy = None  # length n energy vector   : energy by mode
        self.U = None      # M x n x u spatial mode matrix  : spatial modes
        self.V = None      # n x T x u temporal mode matrix : modulate with time

        super().__init__(input_)

    # SKELETON FUNCTIONS: FILL (OVERWRITE) IN SUBCLASS
    def fit_model(self, input_: np.array) -> None:  # skeleton
        """
        Fits the model on the training data: skeleton, overwrite
        :param input_: time series of inputs
        """
        self.compute(input_)

    def get_code(self, input_: np.array) -> np.array: # skeleton
        """
        Passes self.input through the model, returns code
        :input_: time series input
        :return: time series code
        """
        # TODO: Implement this properly; need to get new V, compatible with old mode matrix but new data
        return self.V

    def get_output(self, input_: np.array) -> np.array: # skeleton
        """
        Passes self.code through the model, returns output
        :input_: time series code
        :return: time series output
        """
        self.V = input_
        return self.reconstruct()
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
        eig, phi = LA.eigh(C)

        # Sort Eigenvalues and vectors
        idx = eig.argsort()[::-1]
        eig = eig[idx]
        phi = phi[:, idx]

        # project onto modes for temporal coefficients
        self.V = np.matmul(UU, phi)  # contains the "code" (modal coefficients)

        print(np.shape(phi), (dim[1], dim[2], dim[1] * dim[2], dim[3]))
        self.U = np.reshape(phi, (dim[1], dim[2], dim[1] * dim[2] * dim[3], dim[3]))  # contains the spatial mode

        # contribution of different eigenvectors -> energy
        self.mode_energy = eig / np.sum(eig)

    def reconstruct(self) -> np.array:
        """
        Return the reconstructed flow data from the proper orthogonal decomposition, considering n most energetic modes
        :return: reconstructed flow data
        """
        n = min(self.n, np.shape(self.V)[0])
        return np.matmul(self.V[:, n-1], np.transpose(self.U[:, n-1]))

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


if __name__ == '__main__':
    u_all_1 = POD.preprocess(nu=1, split=False)
    u_all_2 = POD.preprocess(nu=2, split=False)
    Model = POD(u_all_1)
    Model2 = POD(u_all_2)
    Model2.encode(Model2.input)
    Model2.decode(Model2.encoded)
    print(Model2.output)


