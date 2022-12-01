# Libraries
from Models import ClassAE
import time
from csv import DictWriter
import numpy as np
from sklearn.model_selection import ParameterGrid  # pip3.10 -U scikit-learn


param_ranges_dict = {'l_rate': [0.01, 0.001],
                     'epochs': [10, 50, 100],
                     'batch': [1000, 100, 10],
                     'early_stopping': [5, 10, 20],
                     'dimensions': [[8, 4, 2, 1], [16, 8, 4, 2], [24, 12, 6, 3]]}

param_grid = ParameterGrid(param_ranges_dict)  # Flattened grid of all combinations


u_train, u_val, u_test = ClassAE.AE.preprocess()


n = 0
for params in param_grid:
    myModel = ClassAE.AE(**params)
    myModel.u_train = np.copy(u_train)
    myModel.u_val = np.copy(u_val)
    myModel.u_test = np.copy(u_test)
    myModel.input_image()
    myModel.network()
    myModel.creator()
    start = time.time()
    myModel.training()
    end = time.time()
    myModel.performance()

    t_time = end - start

    values = {'Running Time': t_time, 'Loss': myModel.mse
              # , 'Compression': params['dimensions'][-1] / (24 * 24) # this will not generalise well
              }
    values.update(params)

    columns = values.keys()

    print(f'Model {n}')
    n += 1
    with open('TuningDivision/tuning.csv', 'a', newline='') as f:
        writer = DictWriter(f, columns)
        writer.writerow(values)



