from SampleFlows.ParentClass import Model
import numpy as np
from matplotlib import pyplot as plt

train, val, test = Model.preprocess(nu=2)

def conv_mass(grid):

    u_vel = grid[:,:,0]
    v_vel = grid[:,:,1]
    u_vel_grad = np.gradient(u_vel, 1, edge_order=2, axis=0)
    v_vel_grad = np.gradient(v_vel, 1, edge_order=2, axis=1)

    print(np.sum(u_vel_grad))
    print(np.sum(v_vel_grad))

    divergence = u_vel_grad + v_vel_grad

    return np.sum(divergence)

result = conv_mass(train[1,:,:,:])

print(result)


