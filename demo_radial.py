from Main import AE
import numpy as np

data_cartesian = AE.preprocess(nu=2)

# convert
data_radial = []
for set_cartesian in data_cartesian:
    length = np.sqrt(set_cartesian[:, :, :, 0] ** 2 + set_cartesian[:, :, :, 1] ** 2)
    phase = np.arctan(set_cartesian[:, :, :, 1] / set_cartesian[:, :, :, 0])
    set_radial = np.concatenate((length, phase))
    data_radial.append([set_radial])

# fit to data
train, val, test = data_radial
model = AE()
model.fit(train, val)
model.passthrough(test)

# overwrite output (prediction and test to u-v)
prediction = model.y_pred
pred_cart_u = prediction[:, :, :, 0] * np.cos(prediction[:, :, :, 1])
pred_cart_v = prediction[:, :, :, 0] * np.sin(prediction[:, :, :, 1])
pred_cart = np.concatenate((pred_cart_u, pred_cart_v))




# performance metrics
perf = model.performance()
model.verification(test)
model.verification(pred_cart)

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')