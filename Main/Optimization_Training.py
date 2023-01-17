from ClassAE import AE
import time
from csv import DictWriter
from ClassAE import custom_loss_function
from ClassAE import custom_loss_function2
import numpy as np
from sklearn.model_selection import ParameterGrid  # pip3.10 -U scikit-learn

param_ranges_dict = {'l_rate': [0.0001, 0.00001, 0.000001],
                     'epochs': [20, 30, 40],
                     'batch': [50, 20]}

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
    model.fit(custom_loss_function2, u_train, u_val)
    end = time.time()
    model.passthrough(model.u_test)
    model.performance()
    max_div, min_div, avg_div = model.verification(model.y_pred, print_res=False)

    t_time = end - start

    values = {'Loss': model.dict_perf['mse'], 'Divergence': avg_div, 'Absolute' : model.dict_perf['abs_percentage'],
              'Squared': model.dict_perf['sqr_percentage']}

    values.update(params)
    columns = values.keys()

    print(f'Model {k}')

    k += 1

    with open('custom_loss_optimization.csv', 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=columns)
        writer.writerow(values)