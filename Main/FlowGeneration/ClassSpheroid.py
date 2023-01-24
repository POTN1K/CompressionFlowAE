"""
This code presents the class Spheroid that could be used to parametrize the latent space domain. If the spheroid is
plotted in 3D together with the £d scatter plot of the different latent spaces, the similarities become clear. Pay
attention to the fact that theta and phi are defined differently than in the file 'DistanceAnalysis'.
"""

import numpy as np
from FlowGeneration import *


class Spheroid:
    def __init__(self, mode1: list[float], mode2: list[float], mode3: list[float]) -> None:
        """
        Initialize Spheroid object
        :param mode1: list of all the values for the first latent space mode coming from the frame time series
        :param mode2: list of all the values for the second latent space mode coming from the frame time series
        :param mode3: list of all the values for the third latent space mode coming from the frame time series
        """
        self.mode1 = mode1
        self.mode2 = mode2
        self.mode3 = mode3
        self.phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
        self.theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle
        self.center = (stats(mode1)[0], stats(mode2)[0], stats(mode3)[0])
        self.a = stats(mode1)[1]
        self.b = stats(mode2)[1]
        self.c = stats(mode3)[1]

    # BEGIN PROPERTIES
    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value
        self.x = self.a * np.sin(self.theta) * np.cos(self.phi) + self.center[0] * np.ones(np.shape(self.theta))

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value
        self.y = self.b * np.sin(self.theta) * np.sin(self.phi) + self.center[1]

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value
        self.z = self.c * np.cos(self.theta)

    @property
    def distance(self):
        return np.sqrt((self.x - self.center[0]) ** 2 + (self.y - self.center[1]) ** 2 + (self.z - self.center[2]) ** 2)

    # END PROPERTIES

    # BEGIN MODEL METHODS
    def is_inside(self):
        """
        Function to determine how many latent space points lay inside the generated spheroid
        :return: tuple of floats, percentage and number of points inside the spheroid
        """
        positions = np.zeros(np.shape(self.mode1))
        for i in range(len(self.mode1)):
            z_spheroid = self.c ** 2 * (1 - ((self.mode1[i] - self.center[0]) / self.a) ** 2 - (
                        (self.mode2[i] - self.center[1]) / self.b) ** 2)
            k = self.mode3[i] - stats(self.mode3)[0]
            if z_spheroid > k ** 2:
                positions[i] = True
            else:
                positions[i] = False
        points_inside = np.count_nonzero(positions)
        return points_inside / len(positions), points_inside

    def plot_spheroid(self):
        """
        Plots the shperoid together with the latent space points in a 3D plot
        :return: None, plots image
        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel(f'mode 0')
        ax.set_ylabel(f'mode 1')
        ax.set_zlabel(f'mode 2')
        ax.plot_surface(self.x, self.y, self.z)
        ax.scatter3D(self.mode1, self.mode2, self.mode3, '*')
        plt.show()


def dims_analysis(orb: Spheroid):
    """
    Function that modifies the main parameter of the spheroid in order to determine the optimal sizing based on the
    percentage of points inside its volume
    :param orb: object Spheroid
    :return: tuple (float, float, tuple(float, float, float)), first two are the ratios to multiply the parameters of
    the spheroid with in case you want 10% or 90% of the points inside the volume. Last term are the original parameters
    of the spheroid.
    """
    a, b, c = orb.a, orb.b, orb.c
    original = (a, b, c)
    percentage = 0.0
    ratio_max = 1
    while percentage < 0.9:
        orb.a, orb.b, orb.c = ratio_max * a, ratio_max * b, ratio_max * c
        percentage = orb.is_inside()[0]
        ratio_max += 0.1
    ratio_min = 1
    percentage = 1
    while percentage > 0.1:
        orb.a, orb.b, orb.c = ratio_min * a, ratio_min * b, ratio_min * c
        percentage = orb.is_inside()[0]
        ratio_min -= 0.1
    return ratio_min, ratio_max, original


if __name__ == '__main__':
    model = AE.create_trained(2)
    u_all = AE.preprocess(split=False)
    latent = generation_from_original(u_all)
    p1 = latent[:, 0]
    p2 = latent[:, 1]
    p3 = latent[:, 2]
    spheroid = Spheroid(p1, p2, p3)

    # Maximum ratio for having all the points inside (taking 10% outliers) = 1.2
    # Minimum ratio for having all the points outside (taking 10% outliers) = 0.4
