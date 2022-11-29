import numpy as np
from numpy import linalg as LA
from SampleFlows.ParentClass import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class POD(Model):
    def __init__(self, u_all: np.array, n: int = 20, hot_start: int = False) -> None:
        """
        :param u_all: T, i, j, u, v -> time, i_grid, j_grid, vel u, vel v
        :param n: max number of modes to generate
        :param hot_start: bool -> immediately run self.compute
        """
        # TODO: implement attributes from import
        # Parameters
        self.n = n  # max number of modes to consider in generation

        # Dimensions
        # TODO: Finish implementation of dimensions
        self.dim_u = len(np.shape(u_all)) - 3
        if self.dim_u == 1:
            ... # take frm np.shape
        else:
            ...
        # Data
        ...

        if self.n > self.dim_M:  # check validity of n
            raise Exception('Max number of modes n out of range')

        # POD matrices, to be generated
        self.Sigma = None  # Diagonal n x n energy matrix   : energy by mode
        self.U = None      # M x n x u spatial mode matrix  : spatial modes
        self.V = None      # n x T x u temporal mode matrix : modulate with time

        # decoding: _decoded can be generated from variable n modes
        self._decoded = None
        self._n_decoded = None

        # hot start: generate POD matrices
        self.compute()

    def compute(self) -> None:
        """
        Compute and store the sigma, spacial mode, and temporal mode matrices
        :return: None
        """
        raise NotImplemented

    def reconstruct(self) -> np.array:
        """
        Return the reconstructed flow data from the proper orthogonal decomposition
        :return: reconstructed flow data
        """
        raise NotImplemented

    @property
    def decoded(self, n: int = None) -> np.array:
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
    # TODO: Write test
    raise NotImplemented

