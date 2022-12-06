# Libraries
from ClassAE import AE
import time
from csv import DictWriter
import numpy as np

hyperparameters = {'learning rate': [0.01, 0.001],
                   'epochs': [10, 50, 100],
                   'batch': [1000, 100, 10],
                   'early_stopping': [5, 10, 20],
                   'dimensions': [[8, 4, 2, 1], [16, 8, 4, 2], [24, 12, 6, 3]]}

u_train, u_val, u_test = AE.preprocess()
n = 0
for lr in hyperparameters['learning rate']:
    for epoch in hyperparameters['epochs']:
        for batch in hyperparameters['batch']:
            for early_stopping in hyperparameters['early_stopping']:
                for dimensions in hyperparameters['dimensions']:
                    myModel = AE(dimensions=dimensions, l_rate=lr, epochs=epoch, batch=batch,
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
                    with open('TuningDivision/tuning.csv', 'a', newline='') as f:
                        writer = DictWriter(f, columns)
                        writer.writerow(values)
