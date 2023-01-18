# Libraries
import h5py
import numpy as np
import time
from csv import DictWriter
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error  # pip3.10 install scikit-learn NOT sklearn
from sklearn.utils import shuffle
from datetime import datetime
import os
import matplotlib.pyplot as plt


# Generic Model
class Model:
    def __init__(self, train_array: np.array or None = None, val_array: np.array or None = None) -> None:
        self._input = None  # tracks the input array
        self.trained = False  # tracks if the model has been trained
        self._encoded = None  # tracks the encoded array
        self.code_artificial = False  # tracks if the code follows from an input
        self._output = None  # tracks the output array
        self.dict_perf = None  # dictionary of performance for model

        if train_array is not None:  # Hot start
            self.fit(train_array, val_array)

    # BEGIN LOGIC METHODS
    def fit(self, train_array: np.array or None, val_array: np.array or None = None) -> None:
        """
        Train the model on the input data; val_array is optional, see fit_model docstring
        :input_: singular or time series to train the model on
        """
        if train_array is None:  # get stored input
            raise ValueError("Training data not given")

        self.fit_model(train_array, val_array)
        self.trained = True

    def encode(self, input_: np.array) -> np.array:
        """
        Encodes the input array; requires a trained model
        :param input_: singular or time series input
        :return: singular or time series code
        """
        # if not self.trained:
        #     raise Exception('Called encode before fit')

        self.input = input_
        self.encoded = self.get_code(self.input)
        self.code_artificial = False
        return self.encoded

    def decode(self, input_: np.array) -> np.array:
        """
        Returns the decoded input code
        :param input_: singular or time series code
        :return: result of decoding operations, singular or time series depending on input
        """
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
        val_array is optional; required by Keras for training
        :param train_array: time series training data
        :param val_array: optional, time series validation data
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def get_code(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the encoded signal given data to encode
        :input_: time series input
        :return: time series code
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def get_output(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the decoded data given the encoded signal
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
        :return: input time series
        """
        return self._input

    @input.setter
    def input(self, input_: np.array) -> None:
        """
        Overwrite or set model input layer
        :param input_: singular or time series input
        """
        if input_ is not None:
            if len(input_.shape) != 4:
                input_ = np.reshape(input_, (1, *input_.shape))
            self._input = np.copy(input_)

    # TODO: Check compatability with singular inputs (not a priority, but partially implemented)

    @property
    def encoded(self) -> np.array:  # skeleton
        """
        Return the encoded data from the model, calling input as data
        """
        return self._encoded

    @encoded.setter
    def encoded(self, input_: np.array) -> None:
        """
        Sets the encoded attribute to the provided array
        :param input_: code time series
        """
        self._encoded = np.copy(input_)

    @property
    def output(self) -> np.array:
        """
        Returns the output
        :return: copy of the stored output
        """
        output = np.copy(self._output)
        if output.shape[0] == 1:
            output = output[0]
        return output

    @output.setter
    def output(self, input_: np.array) -> None:
        """
        Sets the output
        :param input_: sets the output attribute to the given array
        """
        # if input_.shape[0] == 1:
        #     input_ = input_[0]
        self._output = input_

    # END PROPERTIES

    # BEGIN GENERAL METHODS
    @staticmethod
    def data_reading(re, nx, nu):
        """
        Function to read the H5 files, can change Re to run for different flows
        Re- Reynolds Number
        Nu- Dimension of Velocity Vector
        Nx- Size of grid
        Final dimensions of output: [Time (number of frames), Nx, Nx, Nu]
        """
        # File selection
        # Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0
        # T has different values depending on Re
        if re == 20.0 or re == 30.0 or re == 40.0:
            T = 20000
        else:
            T = 2000

        dir_curr = os.path.split(__file__)[0]
        path_rel = ('SampleFlows', f'Kolmogorov_Re{re}_T{T}_DT01.h5')

        path = os.path.join(dir_curr, *path_rel)

        # READ DATASET
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        # Instantiating the velocities array with zeros
        u_all = np.zeros((nx, nx, len(t), nu))

        # Update u_all with data from file
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
        if nu == 2:
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
    def preprocess(u_all=None, re=40.0, nx=24, nu=2, split=True, norm=True):
        """
        Function to scale the data set and split it into train, validation and test sets.
        nx: Size of the grid side
        nu: Number of velocity components, 1-> 'x', 2 -> 'x','y'
        """

        # Run data reading to avoid errors
        if u_all is None:
            u_all = Model.data_reading(re, nx, nu)

        # Normalize data
        if norm:
            u_min = np.amin(u_all[:, :, :, 0])
            u_max = np.amax(u_all[:, :, :, 0])
            u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
            if nu == 2:
                # Code to run if using velocities in 'y' direction as well
                v_min = np.amin(u_all[:, :, :, 1])
                v_max = np.amax(u_all[:, :, :, 1])
                u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)

        # Division of training, validation and testing data
        if split:
            val_ratio = int(np.round(0.75 * len(u_all)))  # Amount of data used for validation
            test_ratio = int(np.round(0.95 * len(u_all)))  # Amount of data used for testing

            u_train = u_all[:val_ratio, :, :, :].astype('float32')
            u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
            u_test = u_all[test_ratio:, :, :, :].astype('float32')
            return u_train, u_val, u_test
        return u_all

    @staticmethod
    def train_test_batch(param_ranges: dict, model) -> None:
        """
        Trains, evaluates and writes results to file for a model and with hyperparameter ranges
        :param param_ranges: dict with hyperparameters as keys, ranges as items
        :param model: subclass model
        :return: None; results written to timestamped file
        """
        u_train, u_val, u_test = Model.preprocess(nu=2)  # get split data

        param_grid = ParameterGrid(param_ranges)  # Flattened grid of all combinations

        # loop over models
        n = 0
        dir_ = os.path.join(os.path.split(__file__)[0], 'TuningDivision')
        _name = f'_at_{datetime.now().strftime("%m.%d.%Y_%Hh%Mm")}.csv'
        flag = False
        for params in param_grid:
            start_time = time.time()  # get start time

            # initialise model with parameters, specify training set for hot start
            model_ = model(**params, train_array=u_train, val_array=u_val)
            model_.passthrough(u_val)  # sets input and output

            end_time = time.time()  # get end time
            t_time = end_time - start_time
            # compute loss
            perf = model_.performance()

            # write to file
            write = {'Accuracy': perf["abs_percentage"], 'Running Time': t_time, 'Loss': perf["mse"]
                     }
            write.update(params)

            columns = write.keys()
            with open(os.path.join(dir_, f'{model_.__class__.__name__}{_name}'), 'a', newline='') as f:
                writer = DictWriter(f, columns)

                if not flag:  # write column names, its ugly im sorry
                    labels = dict(write)
                    for key in labels.keys():
                        labels[key] = key
                    writer.writerow(labels)
                    flag = True

                writer.writerow(write)  # write results
            print(f'Model {n}')
            n += 1

    @staticmethod
    def verification(data: np.array, print_res: bool = True) -> tuple:
        """
        Function to check conservation of mass
        :param data: time series 2D velocity grid
        :param print_res: bool; true to print results
        :return: max, min, and avg of divergence of velocity with control volume as entire grid
        """

        # List to store values of divergence
        all_conv = []

        for t in range(np.shape(data)[0]):

            # Isolate time components
            grid = data[t, :, :, :]

            # Isolate velocity components
            u_vel = grid[:, :, 0]
            v_vel = grid[:, :, 1]

            # Partial derivatives (du/dx, dv/dy) step size set to 0.262 based on grid size
            u_vel_grad = np.gradient(u_vel, axis=0)
            v_vel_grad = np.gradient(v_vel, axis=1)

            divergence = np.add(u_vel_grad, v_vel_grad)

            all_conv.append(np.sum(divergence))

        max_div = max(all_conv)
        min_div = min(all_conv)
        avg_div = sum(np.abs(all_conv)) / len(all_conv)
        if print_res:
            print(f'max: {max_div}')
            print(f'min: {min_div}')
            print(f'avg: {avg_div}')

        return max_div, min_div, avg_div

    @staticmethod
    def energy(nxnx2: np.array):
        """
        returns the kinetic grid wise energy of one image without taking mass into account
        """
        u = nxnx2[:, :, 0]
        v = nxnx2[:, :, 1]
        return 0.5 * np.add(np.multiply(u, u), np.multiply(v, v))

    @staticmethod
    def curl(nxnx2: np.array):
        """
        returns the curl over the grid of a picture -> curl is used to calculate lift/drag therefore significant
        """
        u = nxnx2[:, :, 0]
        v = nxnx2[:, :, 1]

        return np.subtract(np.gradient(u, axis=1), np.gradient(v, axis=0))

    @staticmethod
    def plot_energy(nxnx2: np.array):
        """
        plots energy/grid without mass/density
        """
        plt.contourf(Model.energy(nxnx2), min=0, max=1.1)
        plt.show()
        return None

    @staticmethod
    def plot_vorticity(nxnx2: np.array):
        """
        This method returns and shows a plot of the cross product of the velocity components
        """
        plt.contourf(Model.curl(nxnx2), min=-2.2, max=2.2)
        plt.show()
        return None

    @staticmethod
    def plot_velocity(nxnx2: np.array):
        """
        plots vectorfield
        """
        x = np.arange(24)
        y = np.arange(24)

        X, Y = np.meshgrid(x, y)

        # Creating plot
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.quiver(X, Y, nxnx2)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        plt.show()
        return None

    @staticmethod
    def u_v_plot(nxnx2):
        """
        Plots velocity components x, y
        :param nxnx2: Time frame for plotting
        :return: None
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.contourf(nxnx2[:, :, 0], vmin=0.0, vmax=1.1)
        ax1.title.set_text('x_velocity')

        ax2 = fig.add_subplot(122)
        ax2.contourf(nxnx2[:, :, 1], vmin=0.0, vmax=1.1)
        ax2.title.set_text('y_velocity')

        fig.suptitle('Velocity Components')
        plt.show()

    @staticmethod
    def plot_all(nxnx2):
        Model.u_v_plot(nxnx2)
        Model.plot_energy(nxnx2)
        Model.plot_vorticity(nxnx2)
        Model.plot_velocity(nxnx2)

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
    # END GENERAL METHODS
