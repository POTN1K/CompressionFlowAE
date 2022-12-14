from Main.ClassAE import AE
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Physical conditions of the flow: Check the values for vorticity (curl), energy, cross product, resultant velocity
# The characteristics of latent space: check changes in latent space due to different Re
domain = np.arange(-0.5, 0.5, 0.05)
model = AE.create_trained()


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


def generation_from_original(n_element):
    """
    n_element: One time frame array, shape = [24,24,2]
    :return: None"""
    # One real element
    u_train, u_val, u_test = AE.preprocess(nu=2)
    test_element = u_test[n_element]

    # Plot original data set
    AE.u_v_plot(test_element)

    latent_space_original = model.encode(test_element)
    print(latent_space_original)
    print(type(latent_space_original))

    reconstructed_original = model.decode(latent_space_original)
    # Plot vorticity
    AE.plot_all(reconstructed_original)


def generate(latent_space):
    """
    Generate artificial flow
    :param latent_space: [m1,m2,m3,m4], values between -1 to 1
    :return: artificial flow [24,24,2]
    """
    artificial = model.decode(np.array([[[latent_space]]]))
    return artificial


if __name__ == '__main__':
    # art_flow = generate([-0.05988009, -0.10334677, 0.45784602, 0.04919271])
    # AE.plot_all(art_flow)
    generation_from_original(0)

