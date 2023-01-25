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
        print(perf['div_avg'])
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


# generate POD data. WARNING if file exists, delete original
generate = False
if generate:
    u_all = POD.preprocess(split=False, nu=2)
    for train_size in [0.95]:
        u_train, u_test = train_test_split(u_all, train_size=train_size)
        gen_val_curve(u_train, u_test)

# load AE data from txt      #bad function, hard coded
load = True
if load:
    path = os.path.join(os.path.split(__file__)[0], 'Main', 'TuningDivision', 'AE_0.95.csv')
    with open(path, newline='')\
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
        flag = False
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

    # load POD data from txt
    path = os.path.join(os.path.split(__file__)[0], 'Main', 'TuningDivision', 'POD_0.95.csv')
    with open(path, newline='')\
            as csvfile:
        rows = reader(csvfile, delimiter=',')
        data_POD = {
            'n': [],
            'mse': [],
            'abs_med': [],
            'abs_mean': [],
            'div_max': [],
            'div_min': [],
            'div_avg': []
        }
        flag = False
        for row in rows:
            if flag:
                mse, abs_med, abs_mean, div_max, div_min, div_avg = row[0], row[1], row[2], row[4], row[5], row[6]
                n = row[-1]
                if n == 'n':
                    raise Exception('Duplicate data in file. Delete file and run generate 1 time to fix')
                lst = [float(n), float(mse), float(abs_med), float(abs_mean), float(div_max), float(div_min),
                       float(div_avg)]
                i = 0
                for key in data_POD.keys():
                    data_POD[key].append(lst[i])
                    i += 1

            if not flag:
                flag = True

plot_abs_med = False
if plot_abs_med:
    if not load:
        raise Exception('load is false')
    # Scatter data
    plt.scatter(data_AE['n'], data_AE['abs_med'], label='AE', color='b', marker='+')
    plt.scatter(data_POD['n'], data_POD['abs_med'], label='POD', color='r', marker='.')

    # Set y axis
    # plt.yscale('log')
    # plt.ylim(bottom=1E-6)
    plt.ylabel('Median Absolute Percentage Accuracy (%)')
    plt.ylim(top=100, bottom=80)


    # Set x axis
    # plt.xscale()
    plt.xlabel('Dimension of Encoded Flow (Orig: 1152)')
    plt.xlim(left=0, right=62.5)

    # Title
    # plt.title('POD and AutoEncoder Accuracy, trained on 3800 flow instances')

    # Plot
    plt.legend()
    plt.show()
    # plt.savefig('POD_AE_abs_med', format='PDF')

plot_med_mean = False
if plot_med_mean:
    if not load:
        raise Exception('load is false')
    # 1st axis
    fig, ax = plt.subplots()
    ax.plot(data_POD['n'], data_POD['abs_med'], label='AE', color='b') #, marker='+')
    ax.set_ylabel('abs_med')

    ax.set_xlabel('Dimension of Encoded Flow (Orig: 1152)')
    ax.set_xlim(left=0, right=62.5)

    # 2nd
    ax2 = ax.twinx()
    ax2.plot(data_POD['n'], data_POD['abs_mean'], label='POD', color='r') #, marker='.')
    ax2.set_ylabel('abs_mean')

    plt.show()

plot_divergence = True
if plot_divergence:
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
