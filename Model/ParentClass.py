# Libraries
import h5py
import numpy as np
import time
from csv import DictWriter
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error  # pip3.10 install scikit-learn NOT sklearn


# Generic Model
class Model:
    def __init__(self, input_: np.array or None = None) -> None:
        self._input = np.copy(input_)  # tracks the input array
        self._encoded = None           # tracks the encoded array
        self._output = None            # tracks the output array

        if input_ is not None:  # Hot start
            self.fit()

    # BEGIN LOGIC METHODS
    def fit(self, input_: np.array or None = None) -> None:
        """
        Train the model, sets the input
        """
        if input_ is None:  # get stored input
            if self.input is None:  # input not specified before fit
                raise ValueError("Input data not found before fit")
            input_ = self.input
        else:  # store input
            self.input = input_

        self.fit_model(input_)

    # TODO: IMPLEMENT ENCODE, DECODE, PASSTHROUGH LOGIC
    # TODO: Check all Logic method logic
    # END LOGIC METHODS

    # SKELETON FUNCTIONS: FILL (OVERWRITE) IN SUBCLASS
    def fit_model(self, input_: np.array) -> None:  # skeleton
        """
        Fits the model on the training data: skeleton, overwrite
        :param input_: time series of inputs
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def get_code(self) -> np.array: # skeleton
        """
        Passes self.input through the model, returns code
        :return: codes time series
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def get_output(self) -> np.array: # skeleton
        """
        Passes self.code through the model, returns output
        :return: output time series
        """
        raise NotImplementedError("Skeleton not filled by subclass")
    # END SKELETONS

    # BEGIN PROPERTIES
    @property
    def input(self) -> np.array:  # skeleton
        """
        Return the input data
        """
        return self._input

    @input.setter
    def input(self, input_: np.array) -> None:
        """
        Overwrite input
        """
        if len(input_.shape) != 4:
            input_ = np.reshape(input_, (1, *input_.shape))
        self._input = np.copy(input_)
        self.code = self.get_code()

    @property
    def encoded(self) -> np.array:  # skeleton
        """
        Return the encoded data from the model, calling input as data
        """
        return self._encoded

    @encoded.setter
    def encoded(self, input_: np.array):
        if len(input_.shape) != 4:
            input_ = np.reshape(input_, (1, *input_.shape))
        self._encoded = np.copy(input_)

    @property
    def output(self):
        """
        Returns the output
        """
        return self.output

    @output.setter
    def output(self, input_):
        """
        Sets the output, removes outer dim for singular result
        """
        if input_.shape[0] == 1:
            input_ = input_[0]
        self.output = input_
    # END PROPERTIES

    # BEGIN GENERAL METHODS
    @staticmethod
    def data_reading(re, nx, nu):
        """Function to read the H5 files, can change Re to run for different flows
            Re- Reynolds Number
            Nu- Dimension of Velocity Vector
            Nx- Size of grid"""
        # FILE SELECTION
        # choose between Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0

        # T has different values depending on Re
        if re == 20.0 or re == 30.0 or re == 40.0:
            T = 20000
        else:
            T = 2000

        path_folder = 'SampleFlows/'  # path to folder in which flow data is situated
        path = path_folder + f'Kolmogorov_Re{re}_T{T}_DT01.h5'

        # READ DATASET
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        u_all = np.zeros((nx, nx, len(t), nu))
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))  # Update u_all with data from file
        if nu == 2:
            u_all[:, :, :, 1] = np.array(hf.get('v_refined'))
        u_all = np.transpose(u_all, [2, 0, 1, 3])  # Time, Nx, Nx, Nu
        hf.close()
        return u_all

    @staticmethod
    def preprocess(u_all=None, re=20.0, nx=24, nu=1, split=True, norm=True):
        if u_all is None:
            u_all = Model.data_reading(re, nx, nu)

        # normalize data
        if norm:
            u_min = np.amin(u_all[:, :, :, 0])
            u_max = np.amax(u_all[:, :, :, 0])
            u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
            if nu == 2:
                v_min = np.amin(u_all[:, :, :, 1])
                v_max = np.amax(u_all[:, :, :, 1])
                u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)

        if split:
            val_ratio = int(np.round(0.75 * len(u_all)))  # Amount of data used for validation
            test_ratio = int(np.round(0.95 * len(u_all)))  # Amount of data used for testing

            u_train = u_all[:val_ratio, :, :, :].astype('float32')
            u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
            u_test = u_all[test_ratio:, :, :, :].astype('float32')
            return u_train, u_val, u_test
        return u_all

    def loss(self) -> float:
        """
        Checks reconstruction loss
        :return: accuracy metric
        """
        # TODO: define proper performance measure
        mse = mean_squared_error(self.input, self.output)
        return mse

    def train_test_batch(self, param_ranges: dict, model) -> None:
        u_train, u_val, u_test = self.preprocess()  # get split data

        param_grid = ParameterGrid(param_ranges)  # Flattened grid of all combinations

        # loop over models
        n = 0
        for params in param_grid:
            start_time = time.time()  # get start time

            # initialise model with parameters, specify training set for hotstart
            model_ = model(**params, input_=u_train)

            end_time = time.time()  # get end time

            # overwrite input, this propagates automatically
            model_.input = u_test

            # compute loss
            loss = self.loss()

            # write to file
            write = {'Running Time': end_time-start_time,
                     'Loss': loss
                     # , 'Compression': params['dimensions'][-1] / (24 * 24) # this will not generalise well
                     }
            write.update(params)

            columns = write.keys()
            with open(f'TuningDivision/tuning{time.time()}.csv', 'a', newline='') as f:
                writer = DictWriter(f, columns)
                writer.writerow(write)
            print(f'Model {n}')
            n += 1
    # END GENERAL METHODS
