# Libraries
import h5py
import numpy as np
import time
from csv import DictWriter
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error  # pip3.10 install scikit-learn NOT sklearn
from sklearn.utils import shuffle
import os

# TODO: finalise docstrings

# Generic Model
class Model:
    def __init__(self, train_array: np.array or None = None, val_array: np.array or None = None) -> None:
        self._input = None  # tracks the input array
        self.trained = False  # tracks if the model has been trained
        self._encoded = None  # tracks the encoded array
        self.code_artificial = False  # tracks if the code follows from an input
        self._output = None  # tracks the output array
        self.dict_perf = None   # dictionary of performance for model

        if train_array is not None:  # Hot start
            self.fit(train_array, val_array)

    # BEGIN LOGIC METHODS
    def fit(self, train_array: np.array or None, val_array: np.array or None = None) -> None:
        """
        Train the model, sets the input
        :input_: singular or time series to train the model on
        """
        if train_array is None:  # get stored input
            raise ValueError("Training data not given")

        self.fit_model(train_array, val_array)
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
        if input_ is not self.encoded:
            self.code_artificial = True
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
    def fit_model(self, train_array: np.array, val_array: np.array or None = None) -> None:  # skeleton
        """
        Fits the model on the training data: skeleton, overwrite
        :param train_array: time series
        :param val_array: optional, time series
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
        if input_ is not None:
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
        self._output = input_
    # END PROPERTIES

    # BEGIN GENERAL METHODS
    @staticmethod
    def data_reading(Re, Nx, Nu):
        """Function to read the H5 files, can change Re to run for different flows
            Re- Reynolds Number
            Nu- Dimension of Velocity Vector
            Nx- Size of grid
            Final dimensions of output: [Time (number of frames), Nx, Nx, Nu]"""
        # File selection
        # Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0
        # T has different values depending on Re
        if Re == 20.0 or Re == 30.0 or Re == 40.0:
            T = 20000
        else:
            T = 2000

        dir_curr = os.path.split(__file__)[0]
        path_rel = ('SampleFlows', f'Kolmogorov_Re{Re}_T{T}_DT01.h5')

        path = os.path.join(dir_curr, *path_rel)

        # READ DATASET
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        # Instantiating the velocities array with zeros
        u_all = np.zeros((Nx, Nx, len(t), Nu))

        # Update u_all with data from file
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
        if Nu == 2:
            u_all[:, :, :, 1] = np.array(hf.get('v_refined'))

        # Transpose of u_all in order to make it easier to work with it
        # New dimensions of u_all = [Time, Nx, Nx, Nu]
        #       - Time: number of frames we have in our data set, which are always related to a different time moment
        #       - Nx: size of the frame in the horizontal component
        #       - Nx: size of the frame in the vertical component
        #       - Nu: dimension of the velocity vector
        u_all = np.transpose(u_all, [2, 0, 1, 3])
        hf.close()

        # Shuffle of the data in order to make sure that there is heterogeneity throughout the test set
        u_all = shuffle(u_all, random_state=42)
        return u_all

    @staticmethod
    def preprocess(u_all=None, Re=40.0, Nx=24, Nu=1):
        """ Function to scale the data set and split it into train, validation and test sets.
            nx: Size of the grid side
            nu: Number of velocity components, 1-> 'x', 2 -> 'x','y'"""

        # Run data reading to avoid errors
        if u_all is None:
            u_all = Model.data_reading(Re, Nx, Nu)

        # Normalize data
        u_min = np.amin(u_all[:, :, :, 0])
        u_max = np.amax(u_all[:, :, :, 0])
        u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
        if Nu == 2:
            # Code to run if using velocities in 'y' direction as well
            v_min = np.amin(u_all[:, :, :, 1])
            v_max = np.amax(u_all[:, :, :, 1])
            u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)

        # Division of training, validation and testing data
        val_ratio = int(np.round(0.75 * len(u_all)))  # Amount of data used for validation
        test_ratio = int(np.round(0.95 * len(u_all)))  # Amount of data used for testing

        u_train = u_all[:val_ratio, :, :, :].astype('float32')
        u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
        u_test = u_all[test_ratio:, :, :, :].astype('float32')
        return u_train, u_val, u_test

    def performance(self) -> dict[str, float]:
        """
        Checks reconstruction loss
        :return: Dictionary
        """
        # TODO: define proper performance measure
        d = dict()
        d['mse'] = mean_squared_error(self.input, self.output)
        self.dict_perf = d
        return d

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
            t_time = end_time-start_time
            # compute loss
            perf = model_.performance()

            # write to file
            write = {'Accuracy': perf['abs_percentage'], 'Running Time': t_time, 'Loss': perf['mse']
                     }
            write.update(params)

            columns = write.keys()
            with open(f'TuningDivision/tuning{time.time()}.csv', 'a', newline='') as f:
                writer = DictWriter(f, columns)
                writer.writerow(write)
            print(f'Model {n}')
            n += 1
    # END GENERAL METHODS