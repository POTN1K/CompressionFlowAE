# Comparison of POD & AE compressive performance
import numpy as np

from FlowCompression import POD
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

        rel_strs = ('FlowCompression', 'TuningDivision', f'POD_{train_size}.csv')
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


# generate POD data. WARNING if file exists, need to delete original
generate = False
load = True
plot_all = True
plot_div = False

if generate:
    u_all = POD.preprocess(split=False, nu=2)
    for train_size in [0.95]:
        u_train, u_test = train_test_split(u_all, train_size=train_size)
        gen_val_curve(u_train, u_test)

# load AE data from txt
if load:
    path = os.path.join(os.path.split(__file__)[0], 'FlowCompression', 'TuningDivision', 'AE_0.95.csv')
    with open(path, newline='')\
            as csvfile:
        rows = reader(csvfile, delimiter=',')
        data_AE = {
            'n': [],
            'mse': [],
            'abs_mean': [],
            'sqr_mean': [],
            'div_max': [],
            'div_min': [],
            'div_avg': []
        }
        flag = False
        for row in rows:
            if flag:
                mse, abs_mean, _, sqr_mean, _, div_max, div_min, div_avg, _, dim, _, _, _ = row
                n = dim.strip('][').split(', ')[3]
                lst = [float(n), float(mse), float(abs_mean), float(sqr_mean), float(div_max), float(div_min),
                       float(div_avg)]
                i = 0
                for key in data_AE.keys():
                    data_AE[key].append(lst[i])
                    i += 1

            if not flag:
                flag = True

    # load POD data from txt
    path = os.path.join(os.path.split(__file__)[0], 'FlowCompression', 'TuningDivision', 'POD_0.95.csv')
    with open(path, newline='')\
            as csvfile:
        rows = reader(csvfile, delimiter=',')
        data_POD = {
            'n': [],
            'mse': [],
            'abs_mean': [],
            'sqr_mean': [],
            'div_max': [],
            'div_min': [],
            'div_avg': []
        }
        flag = False
        for row in rows:
            if flag:
                mse, _, abs_mean, _, sqr_mean, _, _, div_max, div_min, div_avg, n = row
                if n == 'n':
                    raise Exception('Duplicate data in file. Delete file and run generate 1 time to fix')
                lst = [float(n), float(mse), float(abs_mean), float(sqr_mean), float(div_max), float(div_min),
                       float(div_avg)]
                i = 0
                for key in data_POD.keys():
                    data_POD[key].append(lst[i])
                    i += 1

            if not flag:
                flag = True

# plotting
if plot_all:
    data = {'AE': data_AE,
            'POD': data_POD}

    for key in data_POD:
        if key != 'n':
            for label in ['AE', 'POD']:
                if label == 'AE':
                    color = 'b'
                    marker = '+'
                else:
                    color = 'r'
                    marker = '.'

                plt.scatter(data[label]['n'], data[label][key], label=label, color=color, marker=marker)
            plt.ylabel(key)
            plt.xlabel('Dimension of Encoded Flow (Orig: 1152)')
            plt.xlim(left=0, right=62.5)
            plt.legend()
            plt.show()

if plot_div:
    if not load:
        raise Exception('load is false')
    # Scatter data
    # plt.scatter(data_AE['n'], data_AE['div_min'], label='AE', color='b', marker='+')
    # plt.scatter(data_AE['n'], data_AE['div_avg'], label='AE', color='b', marker='+')
    # plt.scatter(data_AE['n'], data_AE['div_max'], label='AE', color='b', marker='+')

    y_err_below = np.abs(np.array(data_POD['div_max'])-np.array(data_POD['div_avg']))
    y_err_above = np.array(data_POD['div_avg'])-np.array(data_POD['div_min'])
    plt.errorbar(data_POD['n'], data_POD['div_avg'], [y_err_above, y_err_below],
                 label='POD', color='black', ecolor='r', marker='.')

    plt.ylabel('Divergence of the Velocity Field')
    plt.ylim(top=2, bottom=-2)

    plt.xlabel('Dimension of Encoded Flow (Orig: 1152)')
    plt.xlim(left=0, right=62.5)

    plt.show()

    y_err_below = np.abs(np.array(data_AE['div_max']) - np.array(data_AE['div_avg']))
    y_err_above = np.array(data_AE['div_avg']) - np.array(data_AE['div_min'])
    plt.errorbar(data_AE['n'], data_AE['div_avg'], [y_err_above, y_err_below],
                 label='AE', color='black', ecolor='b', marker='+')

    plt.ylabel('Divergence of the Velocity Field')
    plt.ylim(top=2, bottom=-2)

    plt.xlabel('Dimension of Encoded Flow (Orig: 1152)')
    plt.xlim(left=0, right=62.5)

    plt.show()
