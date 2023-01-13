from ClassAE import AE
import time
from csv import DictWriter
from ClassAE import custom_loss_function
import numpy as np
from sklearn.model_selection import ParameterGrid  # pip3.10 -U scikit-learn

param_ranges_dict = {'l_rate': [0.01, 0.001, 0.0001],
                     'epochs': [5, 10, 20],
                     'batch': [200, 50, 10]}

param_grid = ParameterGrid(param_ranges_dict)  # Flattened grid of all combinations


u_train, u_val, u_test = AE.preprocess()

k = 0
for params in param_grid:
    n = 2
    u_train, u_val, u_test = AE.preprocess(nu=n)

    model = AE.create_trained()
    model.u_train, model.u_val, model.u_test = u_train, u_val, u_test

    model.epochs = params['epochs']
    model.l_rate = params['l_rate']
    model.batch = params['batch']

    start = time.time()
    model.fit(custom_loss_function, u_train, u_val)
    end = time.time()
    model.passthrough(model.u_test)
    model.performance()
    max_div, min_div, avg_div = model.verification(model.y_pred, print_res=False)

    t_time = end - start

    values = {'Loss': model.dict_perf['mse'], 'Divergence': avg_div
              # , 'Compression': params['dimensions'][-1] / (24 * 24) # this will not generalise well
              }
    values.update(params)

    columns = values.keys()

    print(f'Model {k}')

    k += 1

    with open('custom_loss_optimization.csv', 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=columns)
        writer.writerow(values)