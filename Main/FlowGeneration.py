from Main.ClassAE import AE
import numpy as np

n = 2
u_train, u_val, u_test = AE.preprocess(nu=n)
model = AE.create_trained()
latent_space = np.array([])
#decoder = AE.decode(latent_space)

space = model.encode(u_test)


print(model.decoder.summary())
print(space)