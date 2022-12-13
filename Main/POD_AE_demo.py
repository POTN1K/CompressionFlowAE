# Comparison of POD & AE compressive performance
from Main import AE, POD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# load data AE
# rel_strs = ('TuningDivision', 'iter_3_analysis_2D.csv')
# path = os.path.join(os.getcwd(), *rel_strs)
# data_AE_raw = np.genfromtxt(path, delimiter=',', skip_header=1)
data_AE = {
    64: 0.000483625044580549,
    32: 0.000676902,
    20: 0.0010655912337824702,
    16: 0.,
    14: 0.,
    12: 0.,
    10: 0.,
    8: 0.001975857,
    7: 0.,
    6: 0.,
    5: 0.,
    4: 0.,
    3: 0.003650260390713811,
    2: 0.006444940343499184,
    1: 0.014035881
}
# LATENT SPACE DIM, MSE


# generate POD accuracy for n modes
def gen_val_curve(u_train_, u_test_):
    dim = u_test_.shape
    pod = POD(u_train_)
    y_ = []
    x_ = []

    for n in range(dim[1]*dim[2]):  # max modes n
        pod.n = n

        pod.passthrough(u_test_)
        u_test_mse = np.reshape(u_test_, ([dim[0], dim[1]*dim[2]*dim[3]]))
        output_mse = np.reshape(pod.output, ([dim[0], dim[1]*dim[2]*dim[3]]))

        y_.append(mean_squared_error(u_test_mse, output_mse))
        x_.append(n)
    del pod
    return x_, y_


# actually do the thing
i = 0
cols = ['lightgray', 'darkgray', 'gray', 'black']
for train_size in [0.001, 0.01, 0.1, 0.9]:
    u_all = POD.preprocess(split=False, norm=False)
    u_train, u_test = train_test_split(u_all, train_size=train_size)
    x, y = gen_val_curve(u_train, u_test)
    plt.plot(x, y, label=train_size, color = cols[i])
    i += 1
    print(train_size)


plt.legend()
plt.show()
