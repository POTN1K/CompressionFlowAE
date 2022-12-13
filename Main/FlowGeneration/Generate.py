from Main.ClassAE import AE
n = 2
u_train, u_val, u_test = AE.preprocess(nu=n)
model = AE.create_trained()
print(model.decoder.summary())
latent_space = []
#decoder = AE.decode(latent_space)