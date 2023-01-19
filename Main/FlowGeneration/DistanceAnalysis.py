import matplotlib.pyplot as plt
import numpy as np
import random
from FlowGeneration import *

mode1, mode2, mode3, mode4 = np.transpose(generation_from_original(u_all))
reference_divergence = AE.verification(u_all)[-1]
center = (stats(mode1)[0], stats(mode2)[0], stats(mode3)[0])

# Taking into account all the points present
d = np.sqrt((mode1 - center[0])**2 + (mode2 - center[1])**2 + (mode3 - center[2])**2)[0, 0, :]
phi = np.arctan2(np.sqrt(mode1 ** 2 + mode2 ** 2), mode3)[0, 0, :]
theta = np.arctan2(mode1, mode2)[0, 0, :]
mu = np.mean(d)
sigma = np.std(d)


def create_art():
    d_art = random.gauss(mu, sigma)
    phi_art = random.uniform(0, np.pi)
    theta_art = random.uniform(0, 2 * np.pi)

    m3_art = d_art * np.cos(phi_art)
    m2_art = d_art * np.sin(phi_art) * np.cos(theta_art)
    m1_art = d_art * np.sin(phi_art) * np.sin(theta_art)

    latent_art = [m1_art, m2_art, m3_art, random.uniform(-0.2, 0.2)]

    frame_art = generate(latent_art)

    # Isolate velocity components
    u_vel = frame_art[:, :, 0]
    # Partial derivatives (du/dx, dv/dy) step size set to 0.262 based on grid size
    u_vel_grad = np.gradient(u_vel, axis=0)
    v_vel = frame_art[:, :, 1]
    v_vel_grad = np.gradient(v_vel, axis=1)
    divergence = np.add(u_vel_grad, v_vel_grad)


    return latent_art, np.sum(divergence), frame_art


def thousands(total=1000):
    latents = {}
    for i in tqdm(np.arange(0, total + 1), colour='green'):
        latents[f'{i}'] = create_art()
    return latents


def physicality(interval, latents):
    lower_bound = - interval / 2
    upper_bound = interval / 2
    count = 0
    for i in tqdm(latents, colour='yellow'):
        if lower_bound < latents[i][1] < upper_bound:
            count += 1
    print(f'Rate of physicality {count / len(latents) * 100}% with an interval of {interval}')
    return count / len(latents)


if __name__ == '__main__':
    latents = thousands()
    percentages = []
    for i in tqdm(np.arange(0.2, 2.2, 0.2), colour='blue'):
        percentages.append(physicality(i, latents))
    plt.plot(np.arange(0.1, 1.1, 0.1), percentages)
    plt.xlabel('Divergence Interval')
    plt.ylabel('Rate of Physicality')
    plt.show()









# latent_art, div_art, physical, frame_art = latents['630']
# letter = 'is' if physical else "isn't"
# print(f"The flow {letter} physical")
# print(f'Divergence: {np.sum(div_art)}')
# print(f'Physical accuracy: {count} - {count/total * 100}%')
# AE.u_v_plot(frame_art)


# distance > 0.4 --> 151/200 points are present 75.5%
# distance > 0.35 --> 162/200 points are present 81%
# distance > 0.3 --> 177/200 points are present 88.5%
