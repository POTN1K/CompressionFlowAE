"""
Code to analyse how the magnitude of the latent space modes distribute into the domain. The first three modes are mainly
used because the most important. These are transformed into spherical coordinates (d, phi, theta) and then some known
distribution functions are used to approximate how such quantities spread in the domain. These functions, plus a uniform
distribution for mode 4 allows the team to continuously generate artificial latent space and consequent flow field
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from FlowGeneration import *

# generate a latent space for each frame of the original data set and divide them in modes
mode1, mode2, mode3, mode4 = np.transpose(generation_from_original(u_all))

# calculate center of cluster in order to transform into spherical coords
center = (stats(mode1)[0], stats(mode2)[0], stats(mode3)[0])

# calculate divergence statistics of original frames in order to compare with the artificially generated ones
max_div, min_div, reference_divergence = AE.verification(u_all)


# spherical coordinates transformation
d = np.sqrt((mode1 - center[0])**2 + (mode2 - center[1])**2 + (mode3 - center[2])**2)[0, 0, :]
phi = np.arctan2(np.sqrt(mode1 ** 2 + mode2 ** 2), mode3)[0, 0, :]
theta = np.arctan2(mode1, mode2)[0, 0, :]

# calculate statistics for the distance, needed because gaussian function will be used to approximate its distribution
mu = np.mean(d)
sigma = np.std(d)

plt.hist(d, 20)
plt.show()
plt.hist(phi, 20)
plt.show()
plt.hist(theta, 20)
plt.show()

def plot_scatter(mode1: list[float], mode2: list[float], mode3: list[float]):
    """
    Function to create the scatter plot of three different modes
    :param mode1: list of the magnitudes for the first latent space mode
    :param mode2: list of the magnitudes for the first latent space mode
    :param mode3: list of the magnitudes for the first latent space mode
    :return: None, plots figure
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(mode1, mode2, mode3)
    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('m3')
    plt.show()


def create_art():
    """
    Function that uses distribution functions to generate artificial latent spaces and related flow fields
    :return: artificial latent space, divergence array of artificial flow, boolean checking physical conditions and
             artificial flow field
    """

    # generate values for d, phi, theta with respectively a gaussian, uniform and uniform distribution
    d_art = random.gauss(mu, sigma)
    phi_art = random.uniform(0, np.pi)
    theta_art = random.uniform(0, 2 * np.pi)

    # use artificially generated parameters to determine latent space modes' magnitudes
    m3_art = d_art * np.cos(phi_art)
    m2_art = d_art * np.sin(phi_art) * np.cos(theta_art)
    m1_art = d_art * np.sin(phi_art) * np.sin(theta_art)

    # creation of artificial latent space, mode 4 is found using uniform distribution
    latent_art = [m1_art, m2_art, m3_art, random.uniform(-0.2, 0.2)]

    # generate artifical flow field
    frame_art = generate(latent_art)

    # Isolate velocity components
    u_vel = frame_art[:, :, 0]
    v_vel = frame_art[:, :, 1]
    # Partial derivatives (du/dx, dv/dy) step size set to 0.262 based on grid size
    u_vel_grad = np.gradient(u_vel, axis=0)
    v_vel_grad = np.gradient(v_vel, axis=1)
    divergence = np.add(u_vel_grad, v_vel_grad)

    # check if flow is physical based on a range determined by looking at original time series average divergence
    physicality = min_div < np.sum(divergence) < max_div
    return latent_art, divergence, physicality, frame_art


# Generate 1000 artificial flows in order to compare with original time series
latents: dict[str, tuple[list, list, bool, list]] = {}
count = 0
total = 1000
divergences = []
for i in tqdm(np.arange(0, total), colour='green'):
    latents[f'{i}'] = create_art()
    divergences.append(np.sum(latents[f'{i}'][1]))
    if latents[f'{i}'][2]:
        count += 1

divergences = np.array(divergences)

# print statistics of the artificially generated flow field series
print(np.max(divergences), np.min(divergences), np.sum(np.abs(divergences)) / len(divergences))
print(f'Out of {total} generated frames, {count} are physical. \nPhysical: divergence is between {min_div} and {max_div}.')

