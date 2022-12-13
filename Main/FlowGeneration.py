from Main.ClassAE import AE
import numpy as np

n = 2
# u_train, u_val, u_test = AE.preprocess(nu=n)
model = AE.create_trained()
latent_space = np.array([[[[-1,0,0,0]]]])  # Shape = [None, 1, 1, 4]
artificial_flow = model.decode(latent_space)
print(np.shape(artificial_flow))  # Shape = [24, 24, 2]

plt.subplot(121)
figure1x = plt.contourf(artificial_flow[:, :, 0], vmin=0.0, vmax=1.1)
plt.colorbar(figure1x)
plt.subplot(122)
figure1y = plt.contourf(artificial_flow[:, :, 1], vmin=0.0, vmax=1.1)
plt.colorbar(figure1y)
plt.show()
