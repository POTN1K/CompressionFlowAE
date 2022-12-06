# IMPORT LIBRARIES
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from Main.ParentClass import Model

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------------------------
# READ DATA
u_all = np.concatenate(Model.preprocess(nu=1))
# ------------------------------------------------------------------------------
# COMPUTE POD MODES
Ntrain = 700
dim = u_all.shape
UU = np.reshape(u_all[:Ntrain, :], (Ntrain, dim[1] * dim[2] * dim[3]))

m = UU.shape[0]
C = np.matmul(np.transpose(UU), UU) / (m - 1)

# solve eigenvalue problem
eig, phi = LA.eigh(C)
print(phi.shape)
# Sort Eigenvalues and vectors
idx = eig.argsort()[::-1]
eig = eig[idx]
phi = phi[:, idx]

# project onto modes for temporal coefficients
a = np.matmul(UU, phi)  # contains the "code" (modal coefficients)

phi_spat = np.reshape(phi, (dim[1], dim[2], dim[1] * dim[2]))  # contains the spatial mode

print("check orthogonality")
print(np.matmul(np.transpose(phi), phi))

print("relative contribution of eigenvalues")
contrib = eig / np.sum(eig)

plt.figure()
plt.semilogy(contrib)  # plot the contribution of each mode to the overall energy
plt.show()

# print("size of coefficients")
# print(a.shape)

# print("size of input")
# print(UU.shape)

# ------------------------------------------------------------------------------
# VISUALIZATION OF MODES AND ERROR
imode = 0
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(phi_spat[:, :, imode])

# Visualization of modal reconstruction with truncated number of modes
isample = 50
nmodes = 1

# To reconstruct the field ("decode"), we just matrix-multiply the modal coefficients with the spatial modes
# but we do that for a truncated number of modes, instead of using the full modes
recons = np.matmul(a[:, :nmodes], np.transpose(phi[:, :nmodes]))

fig = plt.figure()
ax = fig.add_subplot(121)
ax.contourf(np.reshape(recons[isample, :], (dim[1], dim[2])))
ax = fig.add_subplot(122)
ax.contourf(np.reshape(UU[isample, :], (dim[1], dim[2])))
plt.show()

# Mean reconstruction error for different number of retained modes
# We can compute the reconstruction of a varying number of modes and compute the error with our original data
err = np.zeros((dim[1] * dim[2]))
for i in range(dim[1] * dim[2]):
    recons = np.matmul(a[:, :i], np.transpose(phi[:, :i]))
    err[i] = np.mean(np.mean(np.square(UU - recons)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(err)
plt.show()
# ------------------------------------------------------------------------------
# PROJECTION OF VAL DATA ONTO TRUNCATED MODE
UU_valid = np.reshape(u_all[Ntrain:, :], (dim[0] - Ntrain, dim[1] * dim[2] * dim[3]))
a_valid = np.matmul(UU_valid, phi)

# Mean reconstruction error for different number of retained modes
err_valid = np.zeros((dim[1] * dim[2]))
for i in range(dim[1] * dim[2]):
    recons = np.matmul(a_valid[:, :i], np.transpose(phi[:, :i]))
    err_valid[i] = np.mean(np.mean(np.square(UU_valid - recons)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(err, 'b')
ax.semilogy(err_valid, 'r')
plt.show()

# Error for n modes
nmodes = 128

print(f'POD error for {nmodes} modes: {err_valid[nmodes]}')
