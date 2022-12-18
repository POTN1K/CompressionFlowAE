# Comparison of POD & AE compressive performance
import numpy as np

from Main import POD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from csv import DictWriter, reader
import numpy

# generate POD accuracy for n modes
def gen_val_curve(u_train_, u_test_):
    dim = u_test_.shape
    pod = POD(u_train_)
    y_ = []
    x_ = []

    flag = False
    for n in range(dim[1]*dim[2]):  # max modes n
        pod.n = n+1

        pod.passthrough(u_test_)
        perf = pod.performance()
        write = {**perf}
        write.update({'n': pod.n})

        columns = write.keys()

        rel_strs = ('Main', 'TuningDivision', f'POD_{train_size}.csv')
        path = os.path.join(os.path.join(os.path.split(__file__)[0], *rel_strs))
        with open(path, 'a', newline='') as f:

            writer = DictWriter(f, columns)

            if not flag:  # write column names, its ugly im sorry
                labels = dict(write)
                for key in labels.keys():
                    labels[key] = key
                writer.writerow(labels)
                flag = True

            writer.writerow(write)  # write results
        # u_test_mse = np.reshape(u_test_, ([dim[0], dim[1]*dim[2]*dim[3]]))
        # output_mse = np.reshape(pod.output, ([dim[0], dim[1]*dim[2]*dim[3]]))
        #
        # y_.append(mean_squared_error(u_test_mse, output_mse))
        # x_.append(n)
    del pod
    return x_, y_


# generate
generate = False
if generate:
    u_all = POD.preprocess(split=False, norm=False, nu=2)
    for train_size in [0.5, 0.95]:
        u_train, u_test = train_test_split(u_all, train_size=train_size)
        gen_val_curve(u_train, u_test)

# load AE data from txt
load = True
if load:
    flag = False
    with open(r'C:\Users\Jan Grobusch\PycharmProjects\CompressionFlowAE\Main\TuningDivision\AE_0.95.csv', newline='')\
            as csvfile:
        rows = reader(csvfile, delimiter=',')
        data_AE = {
            'n': [],
            'mse': [],
            'abs_med': [],
            'div_max': [],
            'div_min': [],
            'div_avg': []
        }
        for row in rows:
            if flag:
                mse, abs_med, div_max, div_min, div_avg = row[0], row[2], row[5], row[6], row[7]
                n = row[9].strip('][').split(', ')[3]
                lst = [float(n), float(mse), float(abs_med), float(div_max), float(div_min), float(div_avg)]
                i = 0
                for key in data_AE.keys():
                    data_AE[key].append(lst[i])
                    i += 1

            if not flag:
                flag = True

    # load AE data from txt
    flag = False
    with open(r'C:\Users\Jan Grobusch\PycharmProjects\CompressionFlowAE\Main\TuningDivision\POD_0.95.csv', newline='')\
            as csvfile:
        rows = reader(csvfile, delimiter=',')
        data_POD = {
            'n': [],
            'mse': [],
            'abs_med': [],
            'div_max': [],
            'div_min': [],
            'div_avg': []
        }
        for row in rows:
            if flag:
                mse, abs_med, div_max, div_min, div_avg = row[0], row[2], row[5], row[6], row[7]
                n = row[-1]
                lst = [float(n), float(mse), float(abs_med), float(div_max), float(div_min), float(div_avg)]
                i = 0
                for key in data_POD.keys():
                    data_POD[key].append(lst[i])
                    i += 1

            if not flag:
                flag = True

plot = True
if plot:
    if not load:
        raise Exception('load is false')
    # Scatter data
    plt.scatter(data_AE['n'], data_AE['mse'], label='AE', color='b', marker='+')
    plt.scatter(data_POD['n'], data_POD['mse'], label='POD', color='r', marker='.')

    # Set y axis
    # plt.yscale('log')
    # plt.ylim(bottom=1E-6)
    plt.ylabel('ylabel')
    plt.ylim(top=100, bottom=0)


    # Set x axis
    # plt.xscale()
    plt.xlabel('Dimension of Encoded Flow (Orig: 1152)')
    plt.xlim(left=0, right=65)

    # Title
    plt.title('Title')

    # Plot
    plt.legend()
    plt.show()
