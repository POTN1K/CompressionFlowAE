import time
from csv import DictWriter
from Main import AE, custom_loss_function
from sklearn.model_selection import ParameterGrid  # pip3.10 -U scikit-learn

param_ranges_dict = {'l_rate': [0.01],
                     'epochs': [10],
                     'batch': [200, 10]}

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

    t_time = end - start

    values = {'Running Time': t_time, 'Loss': model.dict_perf['mse']
              # , 'Compression': params['dimensions'][-1] / (24 * 24) # this will not generalise well
              }
    values.update(params)

    columns = values.keys()

    print(f'Model {k}')

    k += 1

    with open('custom_loss_optimization.csv', 'a', newline='') as f:
        writer = DictWriter(f, columns)
        writer.writerow(values)