from Main.ClassAE import AE
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
u_train, u_val, u_test = AE.preprocess(nu=2)


def generation_from_original(time_series, n_element):
    """
    n_element: One time frame array, shape = [24,24,2]
    :return: None"""
    # One real element
    test_element = time_series[n_element]

    # Plot original data set
    #AE.u_v_plot(test_element)

    latent_space_original = model.encode(test_element)
    return latent_space_original

    #reconstructed_original = model.decode(latent_space_original)
    # Plot vorticity
    #AE.plot_all(reconstructed_original)


def generate(latent_space):
    """
    Generate artificial flow
    :param latent_space: [m1,m2,m3,m4], values between -1 to 1
    :return: artificial flow [24,24,2]
    """
    artificial = model.decode(np.array([[[latent_space]]]))
    return artificial


def generate_from_original_all(time_series):
    if path.exists(f'(0, 1, 2)_latent.csv'):
        latent = np.genfromtxt(f'(0, 1, 2)_latent.csv', delimiter=',')
    else:
        latent = generation_from_original(time_series, 0)[0, 0, 0, :]
        for i in tqdm(range(1, np.shape(time_series)[0]), colour='green'):
            latent = np.vstack((latent, generation_from_original(time_series, i)[0, 0, 0, :]))
    return latent

def original_ls_visual(params: tuple, time_series, plotting=True, saving=False):
    """
    Function to visualiz the first three parameters of all latent spaces of a time seires
    :param params: modes for which we want to visualize the values
    :param plotting: bool to determine is plotting of the results is needed
    :param saving: bool to determine if saving the plot is needed
    :return: None
    """
    if not path.exists(f'{params}_latent.csv'):
        latent = generate_from_original_all(time_series)
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
        p1_inside = p1[inside == True]
        p2_inside = p2[inside == True]
        p3_inside = p3[inside == True]


        ax.plot_surface(x, y, z)
        # Polar
        #r = np.sqrt(p1 ** 2 + p2 ** 2)
        #theta = np.arctan2(p1, p2)
        #print(r)

        ax.scatter3D(p1_inside, p2_inside, p3_inside, '*')
        plt.show()
    if saving is True:
        pickle.dump(fig, open(f'{params}_plot.fig.pickle', 'wb'))


def ranges(params: tuple):
    latent = np.genfromtxt(f'{params}_latent.csv', delimiter=',')
    values = []
    for i in tqdm(range(len(params))):
        values.append((np.max(latent[:, params[i]]), np.min(latent[:, params[i]])))
    print(values)


def average(params: tuple):
    latent = np.genfromtxt(f'{params}_latent.csv', delimiter=',')
    print(np.average(latent[:,0:3]))


def param_analysis(p1, p2, p3):
    # Should return the mean surface for the shape
    ...

def stats(mode: list[float]):
    """
    :param mode: the values of the given mode in all latent spaces of a certain time series
    :return: a list of 5 statistics: midpoint, radius, maximum, minimum, average
    """
    mx = np.max(mode)
    mn = np.min(mode)
    return (mx + mn) / 2, (mx - mn) / 2, mx, mn, np.average(mode)


def spheroid(p1: list[float], p2: list[float], p3: list[float]) -> tuple[list[float], ...]:
    """
    :param p1: the values of this mode in all latent spaces of a certain time series
    :param p2: the values of this mode in all latent spaces of a certain time series
    :param p3: the values of this mode in all latent spaces of a certain time series
    :return: tuple containing the values for x, y, and z which can be used for plotting
    """
    # x**2 / a**2 + y**2 / b**2 + z**2/c**2 = 1
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    a, b, c = stats(p1)[1], stats(p2)[1], stats(p3)[1]
    phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
    theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle
    x = a * np.sin(theta) * np.cos(phi) + stats(p1)[0]
    y = b * np.sin(theta) * np.sin(phi) + stats(p2)[0]
    z = c * np.cos(theta)

    return x, y, z


def is_inside(p1: list[float], p2: list[float], p3: list[float]) -> [list[bool], int]:
    """
    :param p1: the values of this mode in all latent spaces of a certain time series
    :param p2: the values of this mode in all latent spaces of a certain time series
    :param p3: the values of this mode in all latent spaces of a certain time series
    :return:
        - list of bool indicating if point is inside or outside the spheroid
        - number of points inside the spheroid

    """
    positions = np.zeros(np.shape(p1))
    a, b, c = stats(p1)[1], stats(p2)[1], stats(p3)[1]
    for i in range(len(p1)):
        z_spheroid = c ** 2 * (1 - ((p1[i] - stats(p1)[0]) / a) ** 2 - ((p2[i] - stats(p2)[0]) / b) ** 2)
        k = p3[i] - stats(p3)[0]
        if z_spheroid > k ** 2:
            positions[i] = True
        else:
            positions[i] = False
    points_inside = np.count_nonzero(positions)
    return positions, points_inside





if __name__ == '__main__':
    original_ls_visual((0,1,2), u_test, True, False)
