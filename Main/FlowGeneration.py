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


def curl_analysis(k, dictionary):
    """
    Function to get curl for different values of a mode
    k: mode to change
    dictionary: dict to save results
    :return: None
    """
    curls = []
    for i in domain:
        latent_space = np.array([[[[0., 0., 0., 0.]]]])
        latent_space[:, :, :, k] = i
        artificial = model.decode(latent_space)
        curls.append((np.average(model.curl(artificial))))
    dictionary[f'curl_{k}'] = curls


def energy_analysis(k, dictionary):
    """
        Function to get energy for different values of a mode
        k: mode to change
        dictionary: dict to save results
        :return: None
        """
    energy = []
    for i in domain:
        latent_space = np.array([[[[0., 0., 0., 0.]]]])
        latent_space[:, :, :, k] = i
        artificial = model.decode(latent_space)
        energy.append(np.average(model.energy(artificial)))

    dictionary[f'energy_{k}'] = energy


def parameters_analysis_2D():
    """
    :return: plots the parameters of the latent space vs energy or curl
    """
    dictionary = {}
    for j in range(0, 4):
        curl_analysis(j, dictionary)
        energy_analysis(j, dictionary)

    plt.subplot(121)
    plt.plot(domain, dictionary['energy_0'], label='param 1')
    plt.plot(domain, dictionary['energy_1'], label='param 2')
    plt.plot(domain, dictionary['energy_2'], label='param 3')
    plt.plot(domain, dictionary['energy_3'], label='param 4')
    plt.legend()
    plt.title('Energy')
    plt.subplot(122)
    plt.plot(domain, dictionary['curl_0'], label='param 1')
    plt.plot(domain, dictionary['curl_1'], label='param 2')
    plt.plot(domain, dictionary['curl_2'], label='param 3')
    plt.plot(domain, dictionary['curl_3'], label='param 4')
    plt.legend()
    plt.title('Curl')
    plt.show()


def parameters_analysis_3D(mode1, mode2):
    latent_space = np.array([[[[0., 0., 0., 0.]]]])
    p1, p2 = np.meshgrid(domain, domain)
    z_curl = np.zeros(shape=np.shape(p1))
    z_energy = np.zeros(shape=np.shape(p1))
    for i in range(np.shape(domain)[0]):
        for j in range(np.shape(domain)[0]):
            w1 = p1[i, j]
            w2 = p2[i, j]
            latent_space[:, :, :, mode1] = w1
            latent_space[:, :, :, mode2] = w2
            artificial = model.decode(latent_space)
            z_energy[i, j] = np.average(model.energy(artificial))
            z_curl[i, j] = np.average(model.curl(artificial))

    # Plotting
    xy_latent = [0, 0, 0, 0]  # Just needed for title of the plot
    xy_latent[mode1] = 'x'
    xy_latent[mode2] = 'y'

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf_energy = ax1.plot_surface(p1, p2, z_energy, label='energy')
    plt.xlabel(f'x')
    plt.ylabel(f'y')
    ax1.title.set_text('Energy')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_zlim(-0.001, 0.001)
    surf_curl = ax2.plot_surface(p1, p2, z_curl, label='curl')
    plt.xlabel(f'x')
    plt.ylabel(f'y')
    ax2.title.set_text('Curl')
    fig.suptitle(f'Latent Space: {xy_latent}')

    pickle.dump(fig, open(f'{xy_latent}.fig.pickle', 'wb'))


def load_3D_image(mode1, mode2):
    n1, n2 = min(mode1, mode2), max(mode1, mode2)
    xy_latent = [0, 0, 0, 0]  # Just needed for title of the plot
    xy_latent[n2] = 'x'
    xy_latent[n1] = 'y'

    dir_curr = os.path.split(__file__)[0]
    path_rel = ('Plots', f'{xy_latent}.fig.pickle')
    path = os.path.join(dir_curr, *path_rel)
    with open(path, 'rb') as file:
        figx = pickle.load(file)
        plt.show()


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


def original_ls_visual(params: tuple, time_series, plotting=True, saving=False):
    """
    Function to visualiz the first three parameters of all latent spaces of a time seires
    :param params: modes for which we want to visualize the values
    :param plotting: bool to determine is plotting of the results is needed
    :param saving: bool to determine if saving the plot is needed
    :return: None
    """
    if not path.exists(f'{params}_latent.csv'):
        latent = generation_from_original(time_series, 0)[0, 0, 0, :]
        for i in tqdm(range(1, np.shape(time_series)[0]), colour='green'):
            latent = np.vstack((latent, generation_from_original(time_series, i)[0, 0, 0, :]))
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
        ax.plot_surface(x, y, z)
        # Polar
        #r = np.sqrt(p1 ** 2 + p2 ** 2)
        #theta = np.arctan2(p1, p2)
        #print(r)

        ax.scatter3D(p1, p2, p3, '*')
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
    return [(mx + mn) / 2, (mx - mn) / 2, mx, mn, np.average(mode)]


def spheroid(p1: list[float], p2: list[float], p3: list[float]) -> tuple[list[float]]:
    """
    :param p1: the values of this mode in all latent spaces of a certain time series
    :param p2: the values of this mode in all latent spaces of a certain time series
    :param p3: the values of this mode in all latent spaces of a certain time series
    :return: tuple containing the values for x, y, and z which can be used for plotting
    """
    # x**2 / a**2 + y**2 / b**2 + z**2/c**2 = 1
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    a, b, c = stats(p1)[1], stats(p2)[1], stats(p3)[1]
    phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
    theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle
    x = a * np.sin(theta) * np.cos(phi) + stats(p1)[0]
    y = b * np.sin(theta) * np.sin(phi) + stats(p2)[0]
    z = c * np.cos(theta)

    return x, y, z



if __name__ == '__main__':
    original_ls_visual((0,1,2), u_test, True, False)
