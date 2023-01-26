"""
Script to analyze how the reconstructed flow changes due to disturbances to latent space components
"""

from FlowCompression import AE
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

# Import trained model from parent class and draw data from files
model = AE.create_trained(2)
u_all = AE.preprocess(split=False)

# Determine the domain over which the disturbance will be taken
domain = np.arange(-1, 1, 0.1)


# Take one sample from data set and calculate latent space
sample_latent = model.encode(u_all[0])  # [[[[-0.03541018  0.24589653  0.5007892  -0.19181845]]]]

# Plot the original reconstructed flow
AE.u_v_plot(model.decode(sample_latent))

# Define a function to use to analyze the properties of the reconstructed flows (either energy or curl)
def function(frame):
    return AE.curl(frame)

print(np.average(AE.energy(u_all[0])))
print(np.average(AE.curl(u_all[0])))


# For loop to disturbe the latent space, by changing the index of disturbed_latent, you can change which mode is
# disturbed
def disturbance_effects(mode):
    energies = []
    curls = []
    for i in domain:
        # take the sample latent, disturb one of its modes and decode it into an artificial flow
        disturbed_latent = np.copy(sample_latent)
        disturbed_latent[0, 0, 0, mode] = sample_latent[0, 0, 0, mode] * (1 + i)
        img_disturbed = model.decode(disturbed_latent)

        # decode the sample latent and then calculate difference in energy and velocity
        img_sample = model.decode(sample_latent)
        energy_difference = np.subtract(AE.energy(img_disturbed), AE.energy(img_sample))
        img = np.subtract(img_disturbed, img_sample)
        energies.append(np.average(AE.energy(img_disturbed)))
        curls.append(np.average(AE.curl(img_disturbed)))
        # plot the difference in velocity, curl and energy between disturbed and sample flow field
        AE.u_v_curl_plot(img, energy_difference, f'Latent space change   {sample_latent[0][0][0]}   to   {disturbed_latent[0][0][0]}  ')


disturbance_effects(1)
# additional plot of how the energy/curl (depending on function()) changes based on disturbance.
#plt.plot([1 + k for k in domain], energies)
#plt.plot([1 + k for k in domain], curls)
#plt.show()


# -- OBSERVATION FROM PLOTS --

# MODE 1
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
# y velocity becomes positive where it was negative and neg where it was pos,
# increasingly so at higher parameter magnitudes, shifts right a bit
# x velocity waves move right, don't change in intensity or magnitude or direction


# MODE 2
# Hierarchical:
# takes inverse of y-velocity: shifted right and negative
# reverses orientation of 'waves' in x velocity, but keeps the sign / orientation

# - Energy and curl increases
# - X-velocity polarises: top becomes more negative, bottom becomes more positive
# - Y- velocity remains relatively constant


# MODE 3
# Hierarchical:
# not much change in v velocity at all
# only change in x velocity: generally pos and neg areas are flipped in orientation


# average curl increases, 'angle' of the high and low areas in vorticity plots seems to be increasing
# average energy increases quite readily (disregard end values as outside range of Kolmogorov values as parametrized)
# x velocity becomes 'muddied', no real distinct areas can be spotted
# y velocity gets more bundled together and higher in magnitude


# MODE 4
# hierarchical
# almost no change in y velocity
# pos becomes neg and neg becomes pos in x velocity, wave shapes stays same

# energy follows similar trends as previously seen but: shifts a bit to the left as opposed to the right
# again, vorticity first decreases and then increases, seems to shift to the left as with energy
# again, u becomes more muddied and v becomes more 'together', higher magnitudes.
