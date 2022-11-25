import numpy as np
from numpy import linalg as LA
from ParentClass import Model
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class POD(Model):
    def __init__(self, n_train=700, ):
        self.nx = None
        self.nu = None
        self.u_all = None
        self.dim = None
        self.UU = None
        self.n_train = n_train
        self.phi = None
        self.phi_spat = None
        self.contrib = None
        self.a = None
        self.recons = None
        self.err = None

    def computeModes(self, re=20.0, nx=24, nu=1):
        if self.u_all is None:
            self.u_all = POD.data_reading(re, nx, nu)
        self.dim = self.u_all.shape
        self.UU = np.reshape(self.u_all[:self.n_train, :], (self.n_train, self.dim[1] * self.dim[2] * self.dim[3]))
        m = self.UU.shape[0]
        C = np.matmul(np.transpose(self.UU), self.UU) / (m - 1)

        # solve eigenvalue problem
        eig, self.phi = LA.eigh(C)

        # Sort Eigenvalues and vectors
        idx = eig.argsort()[::-1]
        eig = eig[idx]
        self.phi = self.phi[:, idx]

        # project onto modes for temporal coefficients
        self.a = np.matmul(self.UU, self.phi)  # contains the "code" (modal coefficients)

        self.phi_spat = np.reshape(self.phi, (self.dim[1], self.dim[2], self.dim[1] * self.dim[2]))  # contains the spatial mode

        # contribution of different eigenvectors
        self.contrib = eig / np.sum(eig)

    def contribution_visual(self):
        # plot the contribution of each mode to the overall energy

        plt.figure()
        plt.semilogy(self.contrib)
        plt.show()

    def modes_error_visual(self):
        imode = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(self.phi_spat[:, :, imode])

        # Visualization of modal reconstruction with truncated number of modes
        isample = 50
        nmodes = 1

        # To reconstruct the field ("decode"), we just matrix-multiply the modal coefficients with the spatial modes
        # but we do that for a truncated number of modes, instead of using the full modes
        self.recons = np.matmul(self.a[:, :nmodes], np.transpose(self.phi[:, :nmodes]))

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.contourf(np.reshape(self.recons[isample, :], (self.dim[1], self.dim[2])))
        ax = fig.add_subplot(122)
        ax.contourf(np.reshape(self.UU[isample, :], (self.dim[1], self.dim[2])))
        plt.show()

        # Mean reconstruction error for different number of retained modes
        # We can compute the reconstruction of a varying number of modes and compute the error with our original data
        self.err = np.zeros((self.dim[1] * self.dim[2]))
        for i in range(self.dim[1] * self.dim[2]):
            recons = np.matmul(self.a[:, :i], np.transpose(self.phi[:, :i]))
            self.err[i] = np.mean(np.mean(np.square(self.UU - recons)))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy(self.err)
        plt.show()

