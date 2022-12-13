# Comparison of POD & AE compressive performance
from Main import AE, POD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# data AE
data_AE = {
    64: 0.000483625044580549,
    60: 0.000558981264475733,
    56: 0.0003469263028819114,
    52: 0.0007156611536629498,
    48: 0.00040509356767870486,
    44: 0.0004914376186206937,
    40: 0.0005454086931422353,
    36: 0.0011667398503050208,
    32: 0.000676902,
    30: 0.0006799998227506876,
    28: 0.0007047419785521924,
    26: 0.0008112789364531636,
    24: 0.0015978221781551838,
    22: 0.0009955393616110086,
    20: 0.0010655912337824702,
    18: 0.0011272261617705226,
    16: 0.001221104757860303,
    14: 0.001365693868137896,
    12: 0.0014751043636351824,
    10: 0.0016024120850488544,
    8: 0.001975857,
    7: 0.0020148702897131443,
    6: 0.002253078855574131,
    5: 0.0023158101830631495,
    4: 0.0024859101977199316,
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


# generate
generate = False
if generate:
    u_all = POD.preprocess(split=False, norm=False)
    for train_size in [0.95]:
        u_train, u_test = train_test_split(u_all, train_size=train_size)
        x, y = gen_val_curve(u_train, u_test)

        rel_strs = ('TuningDivision', f'POD_{train_size}')
        path = os.path.join(os.getcwd(), *rel_strs)
        np.savetxt(path, np.array([x, y]))

        print(f'{train_size} written to file')

# load POD data from txt
train_size = 0.95
rel_strs = ('TuningDivision', f'POD_{train_size}')
path = os.path.join(os.getcwd(), *rel_strs)
x, y = np.loadtxt(path)
# x = 100 - np.array(x) * 100/(24*24*2)
plt.scatter(x, y, label='POD', color='r', marker='.')

# get AE data from dict
lists = sorted(data_AE.items())  # sort
x, y = zip(*lists)
# x = 100 - np.array(x) * 100/(24*24*2)
plt.scatter(x, y, label='AE', color='b', marker ='+')

plt.xlabel('Dimension of Encoded Flow (Orig: 1152)')
plt.ylabel('MSE')
plt.title('MSE vs Compression for AE and POD, 95% of data used for training')
plt.legend()
plt.show()
