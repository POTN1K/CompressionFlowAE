import numpy as np
from numpy import linalg as LA
from SampleFlows.ParentClass import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class POD(Model):
    def __init__(self, u_all: np.array, n: int = 100, hot_start: int = False) -> None:
        """
        Compatible with 1D or 2D velocity field
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
        self.Sigma = None  # Diagonal n x n energy matrix   : energy by mode
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
        # TODO: port from POD; consider u_all data
        raise NotImplemented

    def reconstruct(self) -> np.array:
        """
        Return the reconstructed flow data from the proper orthogonal decomposition
        :return: reconstructed flow data
        """
        # TODO: reconstruct from POD
        raise NotImplemented

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

    def save_to_file(self) -> None:
        """
        Stretch goal, maybe implement later. Would also need to implement loading properly.
        """
        raise NotImplemented


if __name__ == '__main__':
    u_all = np.concatenate(POD.preprocess(nu=2))
    print(np.shape(u_all))
    print(u_all[0][:][:][0])
    # print(u_all[0])
    # print('......')
    # print(u_all[1])
    Model = POD(u_all, hot_start=True)

