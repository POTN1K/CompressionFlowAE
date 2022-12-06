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
        self.trained = False           # tracks if the model has been trained
        self._encoded = None           # tracks the encoded array
        self.code_artificial = False   # tracks if the code follows from an input
        self._output = None            # tracks the output array

        if input_ is not None:  # Hot start
            self.fit()

    # BEGIN LOGIC METHODS
    def fit(self, input_: np.array or None = None) -> None:
        """
        Train the model, sets the input
        :input_: singular or time series to train the model on
        """
        if input_ is None:  # get stored input
            if self.input is None:  # input not specified before fit
                raise ValueError("Input data not found before fit")
            input_ = self.input
        else:  # store input
            self.input = input_

        self.fit_model(input_)
        self.trained = True

    def encode(self, input_: np.array) -> np.array:
        """
        Encodes the input array with the trained model
        :param input_: singular or time series input
        :return: singular or time series code
        """
        if not self.trained:
            raise Exception('Called encode before fit')

        self.input = input_
        self.encoded = self.get_code(self.input)
        self.code_artificial = False
        return self.encoded

    def decode(self, input_: np.array) -> np.array:
        if not self.trained:
            raise Exception('Called decode before fit')

        self.encoded = input_
        self.output = self.get_output(self.encoded)
        return self.output

    def passthrough(self, input_: np.array) -> np.array:
        """
        Passes the singular or time series input through the encoder and decoder
        Returns the reconstructed form of the input
        :param input_: singular or time series input
        :return: singular or time series output
        """
        return self.decode(self.encode(input_))
    # END LOGIC METHODS

    # SKELETON FUNCTIONS: FILL (OVERWRITE) IN SUBCLASS
    def fit_model(self, input_: np.array) -> None:  # skeleton
        """
        Fits the model on the training data: skeleton, overwrite
        :param input_: time series of inputs
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def get_code(self, input_: np.array) -> np.array: # skeleton
        """
        Passes self.input through the model, returns code
        :input_: time series input
        :return: time series code
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def get_output(self, input_: np.array) -> np.array: # skeleton
        """
        Passes self.code through the model, returns output
        :input_: time series code
        :return: time series output
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
        output = np.copy(self._output)
        if output.shape[0] == 1:
            output = output[0]
        return output

    @output.setter
    def output(self, input_):
        """
        Sets the output
        """
        # if input_.shape[0] == 1:
        #     input_ = input_[0]
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

    @staticmethod
    def train_test_batch(param_ranges: dict, model) -> None:
        """
        Trains, evaluates and writes results to file for a model and with hyperparameter ranges
        :param param_ranges: dict with hyperparameters as keys, ranges as items
        :param model: subclass model
        :return: None; results written to timestamped file
        """
        u_train, u_val, u_test = Model.preprocess()  # get split data

        param_grid = ParameterGrid(param_ranges)  # Flattened grid of all combinations

        # loop over models
        n = 0
        for params in param_grid:
            start_time = time.time()  # get start time

            # initialise model with parameters, specify training set for hot start
            model_ = model(**params, input_=u_train)
            model_.passthrough(u_val)  # sets input and output

            end_time = time.time()  # get end time

            # compute loss
            loss = model_.loss()

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
