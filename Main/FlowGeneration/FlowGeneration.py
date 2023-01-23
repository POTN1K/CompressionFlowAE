from Main import AE
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from os import path
from mpl_toolkits import mplot3d

# Physical conditions of the flow: Check the values for vorticity (curl), energy, cross product, resultant velocity
# The characteristics of latent space: check changes in latent space due to different Re
domain = np.arange(-0.5, 0.5, 0.05)
model = AE.create_trained()
u_all = AE.preprocess(split=False)


def generation_from_original(time_series):
    """
    Generate latent spaces given a certain time series of flows
    :param time_series: series of frame taken from the data set provided by the client
    :return: array of all the latent spaces related to such time frames
    """
    latent_space_original = model.encode(time_series)
    return latent_space_original


def generate(latent_space):
    """
    Generate artificial flow from a given latent space
    :param latent_space: [m1,m2,m3,m4], values between -1 to 1
    :return: artificial flow [24,24,2]
    """
    artificial = model.decode(np.array([[[latent_space]]]))
    return artificial


def original_ls_visual(params: tuple, time_series, plotting=True, saving=False):
    """
    Function to visualize the first three parameters of all latent spaces of a time series
    :param params: modes for which we want to visualize the values
    :param time_series: time series from which to generate the latent spaces
    :param plotting: bool to determine is plotting of the results is needed
    :param saving: bool to determine if saving the plot is needed
    :return: None
    """
    if not path.exists(f'{params}_latent.csv'):
        latent = generation_from_original(time_series)
        np.savetxt(f'{params}_latent.csv', latent, delimiter=',')
    else:
        latent = np.genfromtxt(f'{params}_latent.csv', delimiter=',')
    if plotting is True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.set_xlabel(f'mode {params[0]}')
        ax.set_ylabel(f'mode {params[1]}')
        ax.set_zlabel(f'mode {params[2]}')

        p1 = latent[:, params[0]]
        p2 = latent[:, params[1]]
        p3 = latent[:, params[2]]

        x, y, z = spheroid(p1, p2, p3)
        inside, number = is_inside(p1, p2, p3)
        print(np.size(p1), number)
        p1_inside = p1[inside is True]
        p2_inside = p2[inside is True]
        p3_inside = p3[inside is True]

        ax.plot_surface(x, y, z)

        ax.scatter3D(p1_inside, p2_inside, p3_inside, '*')
        plt.show()
    if saving is True:
        pickle.dump(fig, open(f'{params}_plot.fig.pickle', 'wb'))


def stats(mode: list[float]):
    """
    Determines statistics of a list of values related to one mode of the latent space
    :param mode: the values of the given mode in all latent spaces of a certain time series
    :return: a list of 5 statistics: midpoint, radius, maximum, minimum, average
    """
    mx = np.max(mode)
    mn = np.min(mode)
    return (mx + mn) / 2, (mx - mn) / 2, mx, mn, np.average(mode)


# ------- ANALYSIS OF THE HIERARCHICAL AUTOENCODER ----

def hierarchical_visual(time_series, n_frame):
    """
    Plots the reconstruction effects of the different modes given an initial frame to reconstruct
    :param time_series: time series of frame where to choose from
    :param n_frame: fram number which should be used as a starting point for the reconstruction
    :return: None
    """
    # Generate latent space related to the frame selected
    final_latent = model.encode(u_all[51])[0, 0, 0, :]

    # To see the effect of every mode, every previous component should take the value from 'final_latent' while the
    # next components should be 0
    latent_1 = [final_latent[0], 0, 0, 0]
    latent_2 = [final_latent[0], final_latent[1], 0, 0]
    latent_3 = [final_latent[i] for i in range(3)] + [0]

    # generate flow with each personalized latent. Subtraction is needed to eliminate effects of previous modes
    m1_effect = generate(latent_1)
    m2_effect = generate(latent_2) - m1_effect
    m3_effect = generate(latent_3) - generate(latent_2)
    m4_effect = generate(final_latent) - generate(latent_3)

    # Plot the reconstructed flow and the effects of each mode (different cmap for m2, m3, m4 because both + and -
    AE.u_v_plot(generate(final_latent))
    AE.u_v_plot(m1_effect, title=f'Effect of mode 1 with latent {latent_1}')
    AE.u_v_plot(m2_effect, title=f'Effect of mode 2 with latent {latent_2}', color='seismic')
    AE.u_v_plot(m3_effect, title=f'Effect of mode 3 with latent {latent_3}', color='seismic')
    AE.u_v_plot(m4_effect, title=f'Effect of mode 4 with latent {final_latent}', color='seismic')


if __name__ == '__main__':
    hierarchical_visual(u_all, 51)
