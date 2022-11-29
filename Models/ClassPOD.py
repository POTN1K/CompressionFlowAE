import numpy as np
from numpy import linalg as LA
from SampleFlows.ParentClass import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class POD(Model):
    def __init__(self, u_all: np.array, n_max: int = 20, hot_start: int = False) -> None:
        """
        :param u_all: T, i, j, u, v -> time, i_grid, j_grid, vel u, vel v
        :param n_max: max number of modes to generate
        :param hot_start: bool -> immediately run self.compute
        """
        # TODO: check n int in range

        # TODO: implement attributes from import

        # TODO: implement attributes to be generated
        self._decoded = None
        self._n_decoded = None

        raise NotImplemented

    def compute(self) -> None:
        """
        Compute and
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
                n = self.n_max

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

