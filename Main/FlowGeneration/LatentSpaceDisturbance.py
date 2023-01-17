# Script to analyze how the reconstructed flow changes wìdue to disturbances to latent space components
from Main.ClassAE import AE
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from os import path
from mpl_toolkits import mplot3d

model = AE.create_trained()
u_all = AE.preprocess(u_all=None, re=40.0, nx=24, nu=2, split=False, norm=True)

# Take one sample from data set and calculate latent space
sample_latent = model.encode(u_all[0])
print(sample_latent) # [[[[-0.03541018  0.24589653  0.5007892  -0.19181845]]]]
AE.u_v_plot(u_all[0])
print(np.average(AE.curl(u_all[0])))

# Change mode 1
# energy = []
# for i in [0.005, 0.01, 0.05, 0.075, 0.1]:
#     disturbed_latent = sample_latent
#     disturbed_latent[0, 0, 0, 0] += i
#     print(disturbed_latent)
#     img = model.decode(disturbed_latent)
#     energy.append(np.average(AE.curl(img)))
#     AE.plot_velocity(img)
# plt.plot(np.arange(0,5), energy)
# plt.show()
# Observation:
# - v, u velocity increases but not too much
# - u velocity seems to be creating multiple isolated areas
# - Vorticity goes from negative to positive
# - Energy decreases
# - flow seems becoming more turbulent
# - In vorticity plot, layers become more separated
# - Areas where peak and low energy do not change
# - Flow direction does not change much (arrow plot)
# - v velocity becomes more uniform, u becomes less uniform

# Change mode 2

# energy = [np.average(np.abs(u_all[:,:,1]))]
# for i in [0.01, 0.1, 0.2, 0.5, 0.7]:
#     disturbed_latent = sample_latent
#     disturbed_latent[0, 0, 0, 1] *= 1 + i
#     print(disturbed_latent)
#     img = model.decode(disturbed_latent)
#     energy.append(np.average(np.abs(img[:,:,1])))
#     AE.u_v_plot(img)
# plt.plot([0, 0.01, 0.1, 0.2, 0.5, 0.7], energy)
# plt.show()

# - Energy and curl increases
# - X-velocity polarises: top becomes more negative, bottom becomes more positive
# - Y- velocity remains relatively constant


# Change Mode 3

# energy = [np.average(AE.energy(u_all[0]))]
# for i in [0.01, 0.1, 0.2, 0.5, 0.7]:
#     disturbed_latent = sample_latent
#     disturbed_latent[0, 0, 0, 2] *= 1 + i
#     print(disturbed_latent)
#     img = model.decode(disturbed_latent)
#     energy.append(np.average(AE.energy(img)))
#     AE.u_v_plot(img)
# plt.plot([0, 0.01, 0.1, 0.2, 0.5, 0.7], energy)
# plt.show()

# average curl increases, 'angle' of the high and low areas in vorticity plots seems to be increasing
# average energy increases quite readily (disregard end values as outside range of Kolmogorov values as parametrized)
# x velocity becomes 'muddied', no real distinct areas anymore
# y velocity gets more bundled together and higher in magnitude

# Mode 4

energy = [np.average(AE.curl(u_all[0]))]
for i in [0.01, 0.1, 0.2, 0.5, 0.7]:
    disturbed_latent = sample_latent
    disturbed_latent[0, 0, 0, 3] *= 1 + i
    print(disturbed_latent)
    img = model.decode(disturbed_latent)
    energy.append(np.average(AE.curl(img)))
    AE.u_v_plot(img)
plt.plot([0, 0.01, 0.1, 0.2, 0.5, 0.7], energy)
plt.show()

# energy follows similar trends as previously seen but: shifts a bit to the left as opposed to the right
# again, vorticity first decreases and then increases, seems to shift to the left as with energy
# again, u becomes more muddied and v becomes more 'together', higher magnitudes.