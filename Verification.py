from SampleFlows.ParentClass import Model
import numpy as np
from matplotlib import pyplot as plt
import math

# LOAD DATA
train, val, test = Model.preprocess(nu=2)

def conv_mass(grid_time, t):
    ''' Function to check conservation of mass
    grid_time: time series 2D velocity grid
    t: time
    output: divergence of velocity with control volume as entire grid'''

    # Isolate time components
    grid = grid_time[t,:,:,:]

    #Isolate velocity components
    u_vel = grid[:,:,0]
    v_vel = grid[:,:,1]

    # Partial derivatives (du/dx, dv/dy)
    u_vel_grad = np.gradient(u_vel, 0.262, edge_order=2, axis=1)
    v_vel_grad = np.gradient(v_vel, 0.262, edge_order=2, axis=0)

    divergence = u_vel_grad + v_vel_grad

    # Optional plotting of divergence
    # plt.contourf(divergence)
    # plt.show()

    return np.sum(divergence)


all_conv = []

for time in range(np.shape(train)[0]):
    all_conv.append(conv_mass(train, time))

print('max', max(all_conv))
print('min', min(all_conv))
print('avg', sum(all_conv) /len(all_conv))


# def navier_stokes(grid_time, t):
#     u = grid_time[:, :, :, 0]
#     v = grid_time[:, :, :, 1]
#
#     du_dx = np.gradient(u, 1, edge_order=2, axis=2)
#     du_dy = np.gradient(u, 1, edge_order=2, axis=1)
#     dv_dx = np.gradient(v, 1, edge_order=2, axis=2)
#     dv_dy = np.gradient(v, 1, edge_order=2, axis=1)
#
#     du_dt = np.gradient(u, 1, edge_order=2, axis=0)
#     dv_dt = np.gradient(v, 1, edge_order=2, axis=0)
#
#     d2u_d2x = np.gradient(du_dx, 1, edge_order=2, axis=2)
#     d2v_d2y = np.gradient(dv_dy, 1, edge_order=2, axis=1)
#
#     velocity_2 = np.dot(u[t,:,:], v[t,:,:]) * 0.5
#
#     dvelocity_2_dx = np.gradient(velocity_2, 1, edge_order=2, axis=1)
#     dvelocity_2_dy = np.gradient(velocity_2, 1, edge_order=2, axis=0)
#
#     def convection(u, du_dx, v, du_dy, dy_dx, dv_dy):
#         return np.array([np.add(np.multiply(u, du_dx), np.multiply(v, du_dy)),
#                         np.add(np.multiply(u, dv_dx), np.multiply(v, dv_dy))])
#
#     def diffusion(d2u_d2x, d2u_d2y, d2v_d2x, d2v_d2y):
#         return np.array([np.add(d2u_d2x, d2u_d2y),
#                          np.add(d2v_d2x, d2v_d2y)])
#
#     def internal(dvelocity_2_dx, dvelocity_2_dy):
#         return -1 * np.array([dvelocity_2_dx, dvelocity_2_dy])
#
#     def temporal(du_dt, dv_dt):
#         return np.array(du_dt, dv_dt)
#
#     def force():
#         temp = [[math.sin(4 * 2 * np.pi * i / (24)), 0] for i in range(24)]
#         new = np.array([temp for i in range(24)])
#
#         return np.moveaxis(new, 0, 1)

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
