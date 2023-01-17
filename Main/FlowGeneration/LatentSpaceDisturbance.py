# Script to analyze how the reconstructed flow changes due to disturbances to latent space components
from Main.ClassAE import AE
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from os import path
from mpl_toolkits import mplot3d

domain = np.arange(-0.5, 0.5, 0.05)
model = AE.create_trained(h=True)
u_all = AE.preprocess(nu=2, split=False)
values = []

# Take one sample from data set and calculate latent space
sample_latent = model.encode(u_all[0])
range = np.arange(-2.5, 0.7, 0.1)
print(sample_latent)  # [[[[-0.03541018  0.24589653  0.5007892  -0.19181845]]]]
AE.u_v_plot(model.decode(sample_latent))


def function(input):
    return AE.curl(input)


print(np.average(function(u_all[0])))

# Change mode 1
for i in range:
    disturbed_latent = np.copy(sample_latent)
    print(1 + i)
    print(sample_latent[0, 0, 0, 3])
    disturbed_latent[0, 0, 0, 3] = sample_latent[0, 0, 0, 3] * (1 + i)
    print(disturbed_latent)
    img = model.decode(disturbed_latent)
    values.append(np.average(function(img)))
    AE.u_v_plot(img)
plt.plot([k + 1.0 for k in range], values)
plt.show()
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

# hierarchical:
# as goes from -1.5 times itself to +1.5:
# y velocity becomes positive where it was negatve and neg where it was pos,
# increasingly so at higher parameter magnitudes, shifts right a bit
# x velocity waves move right, don't change in intensity or magnitude or direction


# Change mode 2

#Hierarchical:
# takes inverse of y-velocity: shifted right and negative
# reverses orientation of 'waves' in x velocity, but keeps the sign / orientation

# - Energy and curl increases
# - X-velocity polarises: top becomes more negative, bottom becomes more positive
# - Y- velocity remains relatively constant


# Change Mode 3
#Hierarchical:
# not much change in v velocity at all
# only change in x velocity: generally pos and neg areas are flipped in orientation


# average curl increases, 'angle' of the high and low areas in vorticity plots seems to be increasing
# average energy increases quite readily (disregard end values as outside range of Kolmogorov values as parametrized)
# x velocity becomes 'muddied', no real distinct areas anymore
# y velocity gets more bundled together and higher in magnitude

# Mode 4

# hierarchical
# almost no change in y velocity
# pos becomes neg and neg becomes pos in x velocity, wave shapes stays same

# energy follows similar trends as previously seen but: shifts a bit to the left as opposed to the right
# again, vorticity first decreases and then increases, seems to shift to the left as with energy
# again, u becomes more muddied and v becomes more 'together', higher magnitudes.