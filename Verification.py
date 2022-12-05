from SampleFlows.ParentClass import Model
import numpy as np
from matplotlib import pyplot as plt

train, val, test = Model.preprocess(nu=2)

def conv_mass(grid_time, t):

    grid = grid_time[t,:,:,:]

    u_vel = grid[:,:,0]
    v_vel = grid[:,:,1]

    u_vel_grad = np.gradient(u_vel, 1, edge_order=2, axis=1)
    v_vel_grad = np.gradient(v_vel, 1, edge_order=2, axis=0)

    divergence = u_vel_grad + v_vel_grad

    # plt.contourf(divergence)
    # plt.show()

    return np.sum(divergence)

def navier_stokes(grid_time, t):
    u = grid_time[:, :, :, 0]
    v = grid_time[:, :, :, 1]

    du_dx = np.gradient(u, 1, edge_order=2, axis=2)
    dv_dy = np.gradient(v, 1, edge_order=2, axis=1)

    du_dt = np.gradient(u, 1, edge_order=2, axis=0)
    dv_dt = np.gradient(v, 1, edge_order=2, axis=0)

    d2u_d2x = np.gradient(du_dx, 1, edge_order=2, axis=2)
    d2v_d2y = np.gradient(dv_dy, 1, edge_order=2, axis=1)

    velocity_2 = np.dot(u[t,:,:], v[t,:,:])

    dvelocity_2_dx = np.gradient(velocity_2, 1, edge_order=2, axis=1)
    dvelocity_2_dy = np.gradient(velocity_2, 1, edge_order=2, axis=0)

    

navier_stokes(train, 100)


# Testing for np.gradient function
#
# a = np.arange(0,10, 1)
# b = a * a
# b = np.array([[b,b,b,b,b], [b,b,b,b,b], [b,b,b,b,b], [b,b,b,b,b]])
# b = b
#
# print(np.shape(b))
#
# print(np.gradient(b, 1, edge_order=2, axis=2))
