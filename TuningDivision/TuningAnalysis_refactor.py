# Code to tune any type of autoencoder. It will take ranges for multiple hyperparameters and output
# one model for each set of those

# Libraries
import time
from csv import DictWriter
import numpy as np
from sklearn.model_selection import ParameterGrid  # pip3.10 -U scikit-learn
# Local codes
from Models import ClassAE


# Ranges for the hyperparameters that have to be tuned
param_ranges_dict = {'l_rate': [0.0005, 0.0001, 0.00005],
                     'epochs': [500],
                     'batch': [10],
                     'early_stopping': [10],
                     'dimensions': [[256, 128, 64, 32]],
                     'Nu': [2]}

# Flattened grid of all combinations of hyperparameters
param_grid = ParameterGrid(param_ranges_dict)

# Data reading and preprocess before starting the tuning to save computational time
u_train, u_val, u_test = ClassAE.AE.preprocess(Nu=2)


n = 0
for params in param_grid:
    myModel = ClassAE.AE(**params)
    myModel.u_train = np.copy(u_train)
    myModel.u_val = np.copy(u_val)
    myModel.u_test = np.copy(u_test)
    myModel.network()
    myModel.creator()
    start = time.time() # Time calculation to train the model
    myModel.training()
    end = time.time()
    myModel.performance()

    t_time = end - start

    # Define values to append to the csv file. These will be:
    #   -'Running Time' and 'Loss': important for accuracy analysis
    #   - hyperparameters: important to replicate the most optimal models

    values = {'Accuracy': myModel.abs_percentage, 'Running Time': t_time, 'Loss': myModel.mse
              # , 'Compression': params['dimensions'][-1] / (24 * 24) # this will not generalise well
              }
    values.update(params)

    columns = values.keys()

    # Keep track of number of models generated
    print(f'Model {n}')
    n += 1

    # Append to csv
    with open('tuning.csv', 'a', newline='') as f:
        writer = DictWriter(f, columns)
        writer.writerow(values)



