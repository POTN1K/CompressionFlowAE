from FlowGeneration import *
domain = np.arange(-0.5, 0.5, 0.05)
model = AE.create_trained()
u_train, u_val, u_test = AE.preprocess(nu=2)

class Spheroid:
    def __init__(self, mode1: list[float], mode2: list[float], mode3: list[float]) -> None:
        self.a = stats(mode1)[1]
        self.b = stats(mode2)[1]
        self.c = stats(mode3)[1]
        self.center = (stats(mode1)[0], stats(mode2)[0], stats(mode3)[0])
        self.phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
        self.theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle
        self.x = None
        self.y = None
        self.z = None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

    def surface_coordinates(self) -> tuple[list[float], list[float], list[float]]:
        self.x = self.a * np.sin(self.theta) * np.cos(self.phi) + self.center[0]
        self.y = self.b * np.sin(self.theta) * np.sin(self.phi) + self.center[1]
        self.z = self.c * np.cos(self.theta)
        return self.x, self.y, self.z

    def is_inside(self, mode1, mode2, mode3):
        positions = np.zeros(np.shape(mode1))
        for i in range(len(mode1)):
            z_spheroid = self.c ** 2 * (1 - ((mode1[i] - self.center[0]) / self.a) ** 2 - ((mode2[i] - self.center[1]) / self.b) ** 2)
            k = mode3[i] - stats(mode3)[0]
            if z_spheroid > k ** 2:
                positions[i] = True
            else:
                positions[i] = False
        points_inside = np.count_nonzero(positions)
        return positions, points_inside



def run():
    p1, p2, p3 = generate_from_original_all(u_test)
    spheroid = Spheroid(p1,p2,p3)