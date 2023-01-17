from Main import AE
import numpy as np

data_cartesian = AE.preprocess(nu=2)

# convert
data_radial = []
for set_cartesian in data_cartesian:
    length = np.sqrt(set_cartesian[:, :, :, 0] ** 2 + set_cartesian[:, :, :, 1] ** 2)
    out = np.sign(set_cartesian[:, :, :, 0]) * np.pi / 2
    phase = np.arctan(np.divide(set_cartesian[:, :, :, 1], set_cartesian[:, :, :, 0], out,
                                where=set_cartesian[:, :, :, 0] != 0))
    set_radial = np.array([length, phase])
    set_radial = np.moveaxis(set_radial, 0, -1)
    print(set_radial.shape)
    data_radial.append(set_radial)

# fit to data
train, val, test = data_radial
model = AE()
model.fit(train, val)
model.passthrough(test)

# overwrite output (prediction to u-v)
prediction = model.y_pred
pred_cart_u = prediction[:, :, :, 0] * np.cos(prediction[:, :, :, 1])
pred_cart_v = prediction[:, :, :, 0] * np.sin(prediction[:, :, :, 1])
pred_cart = np.array([pred_cart_u, pred_cart_v])
pred_cart = np.moveaxis(pred_cart, 0, -1)


# performance metrics
perf = model.performance()
model.verification(data_cartesian[2])
model.verification(pred_cart)

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

# Save model
model.autoencoder.save('autoencoder_Polar.h5')
model.encoder.save('encoder_Polar.h5')
model.decoder.save('decoder_Polar.h5')