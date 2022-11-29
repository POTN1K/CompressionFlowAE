# Libraries
from Models import ClassAE
import time
from csv import DictWriter
import numpy as np
import os

filename = 'tuning_2.csv'
hyperparameters = {'learning rate': [0.0001, 0.0005],
                   'epochs': [80, 100, 200],
                   'batch': [20, 10, 5],
                   'early_stopping': [10],
                   'dimensions': [[24, 12, 6, 3], [16, 8, 4, 2], [32, 16, 8, 4], [64, 32, 16, 8], [256, 128, 64, 32]]}

u_train, u_val, u_test = ClassAE.AE.preprocess()
n = 0
for lr in hyperparameters['learning rate']:
    for epoch in hyperparameters['epochs']:
        for batch in hyperparameters['batch']:
            for early_stopping in hyperparameters['early_stopping']:
                for dimensions in hyperparameters['dimensions']:
                    myModel = ClassAE.AE(dimensions=dimensions, l_rate=lr, epochs=epoch, batch=batch,
                                         early_stopping=early_stopping)
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
                    columns = ['Running Time', 'Loss', 'Compression', 'Learning Rate', 'Epochs', 'Batch Size',
                               'Early Stopping', 'Dimensions']

                    values = {'Running Time': t_time, 'Loss': myModel.mse, 'Compression': dimensions[-1] / (24 * 24),
                              'Learning Rate': lr, 'Epochs': epoch, 'Batch Size': batch,
                              'Early Stopping': early_stopping,
                              'Dimensions': dimensions}
                    print(f'Model {n}')
                    n += 1
                    with open('tuning.csv', 'a', newline='') as f:
                        writer = DictWriter(f, columns)
                        writer.writerow(values)
