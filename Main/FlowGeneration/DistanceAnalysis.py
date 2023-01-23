import matplotlib.pyplot as plt
import numpy as np
import random
from FlowGeneration import *

mode1, mode2, mode3, mode4 = np.transpose(generation_from_original(u_all))
max_div, min_div, reference_divergence = AE.verification(u_all)
center = (stats(mode1)[0], stats(mode2)[0], stats(mode3)[0])

# Taking into account all the points present
d = np.sqrt((mode1 - center[0])**2 + (mode2 - center[1])**2 + (mode3 - center[2])**2)[0, 0, :]
phi = np.arctan2(np.sqrt(mode1 ** 2 + mode2 ** 2), mode3)[0, 0, :]
theta = np.arctan2(mode1, mode2)[0, 0, :]
mu = np.mean(d)
sigma = np.std(d)

def plot_scatter(mode1, mode2, mode3):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(mode1, mode2, mode3)
    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('m3')
    plt.show()


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
    v_vel = frame_art[:, :, 1]
    # Partial derivatives (du/dx, dv/dy) step size set to 0.262 based on grid size

    u_vel_grad = np.gradient(u_vel, axis=0)
    v_vel_grad = np.gradient(v_vel, axis=1)
    divergence = np.add(u_vel_grad, v_vel_grad)
    physicality = reference_divergence * 0.1 < np.sum(divergence) < reference_divergence * 1.9
    return latent_art, divergence, physicality, frame_art


latents = {}
count = 0
total = 1000
divergences = []
for i in tqdm(np.arange(0, total + 1), colour='green'):
    latents[f'{i}'] = create_art()
    divergences.append(np.sum(latents[f'{i}'][1]))
    if latents[f'{i}'][2]:
        count += 1

divergences = np.array(divergences)
print(np.max(divergences), np.min(divergences), np.sum(np.abs(divergences)) / len(divergences))
print(count)

# domain = np.arange(0.1, 1.1, 0.1)
# ratios = []
# for i in domain:
#     count = 0
#     for j in latents:
#         physicality = reference_divergence * (1 - i) < np.sum(latents[j][1]) < reference_divergence * (1 + i)
#         if physicality:
#             count += 1
#     print(count)
#     ratios.append(count / len(latents) * 100)

# plt.plot(domain, ratios)
# plt.show()
# at 1.25: 16.4%
# at 1.5: 25.7%

