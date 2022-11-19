# IMPORT LIBRARIES
import os
import numpy as np
import argparse
from numpy import linalg as LA
import matplotlib.pyplot as plt
import h5py

# ------------------------------------------------------------------------------
# READ DATASET
hf = h5py.File('/content/drive/MyDrive/Kolmogorov_Re10_T20000_DT01.h5', 'r')
print(hf)
Nx = 24
Nu = 1
t = np.array(hf.get('t'))
u_all = np.zeros((Nx, Nx, len(t), Nu))
u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
# if Nu ==2:
#     u_all[:,:,:,1] = np.array(hf.get('v_refined'))
u_all = np.transpose(u_all, [2, 0, 1, 3])
hf.close()
print(u_all.shape)

# normalize data
u_min = np.amin(u_all[:, :, :, 0])
u_max = np.amax(u_all[:, :, :, 0])
u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
# if Nu==2:
#     v_min = np.amin(u_all[:,:,:,1])
#     v_max = np.amax(u_all[:,:,:,1])
#     u_all[:,:,:,1] = (u_all[:,:,:,1] - v_min) / (v_max - v_min)

# ------------------------------------------------------------------------------
# COMPUTE POD MODES
Ntrain = 700
dim = u_all.shape

UU = np.reshape(u_all[:Ntrain, :], (Ntrain, dim[1] * dim[2] * dim[3]))

m = UU.shape[0]
C = np.matmul(np.transpose(UU), UU) / (m - 1)

# solve eigenvalue problem
eig, phi = LA.eigh(C)

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

# Mean reconstruction error for different number of retained modes
# We can compute the reconstruction of a varying number of modes and compute the error with our original data
err = np.zeros((dim[1] * dim[2]))
for i in range(dim[1] * dim[2]):
    recons = np.matmul(a[:, :i], np.transpose(phi[:, :i]))
    err[i] = np.mean(np.mean(np.square(UU - recons)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(err)

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

# Error for n modes
nmodes = 128

print(f'POD error for {nmodes} modes: {err_valid[nmodes]}')
