""" Generic Model for Flow Compression Techniques

This file defines a class "Model" which is used as a skeleton for the implementation of
different autoencoders and principal orthogonal decomposition.

The class's methods are divided into:
    * Logic Methods - Called by the user, used to work with the model in a high level
    * Skeleton functions - Used in subclasses. Handle errors and are lower level
    * General methods - Static methods to generate models, or assess shared characteristics of models
"""

# Libraries
# Utils
import os
import time
from datetime import datetime
# File management
from csv import DictWriter
import h5py
# Plot
import matplotlib.pyplot as plt
# Numerics
import numpy as np
# ScikitLearn -> pip3.10 install scikit-learn NOT sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle


# Generic Model
class Model:
    """
    Generic Model used as a parent class for flow compression, it works as a skeleton
    for the subclasses, it also handles exceptions.
    """

    def __init__(self, train_array: np.array or None = None, val_array: np.array or None = None) -> None:
        """
        Initialize Model object
        :param train_array: numpy array, optional, if provided then the model is hot-started and fit on this array
        :param val_array: numpy array, optional, validation data for the training process
        :return: None
        """

        # Attributes used throughout the code
        self._input = None  # tracks the input array
        self._encoded = None  # tracks the encoded array
        self._output = None  # tracks the output array
        self.dict_perf = None  # dictionary of performance for model

        if train_array is not None:  # Hot start - Start training immediately
            self.fit(train_array, val_array)

    # BEGIN PROPERTIES
    @property
    def input(self) -> np.array:
        """
        Return the input data
        :return: numpy array, input time series
        """
        return self._input

    @input.setter
    def input(self, input_: np.array) -> None:
        """
        Set model input or overwrite shape.
        All models work with 4D arrays
        :param input_:  numpy array, singular or time series input
        :return: None
        """
        if input_ is not None:
            if len(input_.shape) != 4:
                input_ = np.reshape(input_, (1, *input_.shape))
            self._input = np.copy(input_)

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
        :return: None
        """
        # if input_.shape[0] == 1:
        #     input_ = input_[0]
        self._output = input_

    @property
    def encoded(self) -> np.array:  # skeleton
        """
        Generic variable of a singular or time series latent space
        :return:  numpy array, latent space
        """
        return self._encoded

    @encoded.setter
    def encoded(self, input_: np.array) -> None:
        """
        Saves a copy of the latent space
        :param input_: code time series
        :return: None
        """
        self._encoded = np.copy(input_)

    # END PROPERTIES

    # BEGIN LOGIC METHODS
    # High level methods used by the user
    def fit(self, train_array: np.array, val_array: np.array or None = None) -> None:
        """
        Train the model on the input data
        :param train_array: numpy array, used to train the model
        :param val_array: numpy array, optional, depending on the model it will need a validation set
        :return: None
        """

        self._fit_model(train_array, val_array)

    def encode(self, input_: np.array) -> np.array:
        """
        Encodes the input array using the model
        :param input_: numpy array, singular or time series input
        :return: numpy array, singular or time series latent space
        """

        self.input = input_
        self.encoded = self._get_code(self.input)
        return self.encoded

    def decode(self, input_: np.array) -> np.array:
        """
        Returns the decoded input code using the model
        :param input_: numpy array, singular or time series latent space. Size depends on each model
        :return: numpy array, singular or time series depending on input
        """

        self.encoded = input_
        self.output = self._get_output(self.encoded)
        return self.output

    def passthrough(self, input_: np.array) -> np.array:
        """
        Passes the singular or time series input through the encoder and decoder
        Returns the reconstructed form of the input
        :param input_: numpy array, singular or time series input
        :return: numpy array, singular or time series output
        """

        return self.decode(self.encode(input_))

    def performance(self) -> dict[str, float]:
        """
        Creates a dictionary with general metrics for measuring the accuracy of the model
        :return: Dictionary with relevant accuracy metrics
        """
        d = dict()
        d['mse'] = np.mean((self.output-self.input)**2)

        # Absolute percentage metric
        percentage = 100 * (1 - (np.abs((self.input - self.output) / self.input)))  # get array; we use it 3 times
        d['abs_median'] = np.median(percentage)
        d['abs_mean'] = np.mean(percentage)  # tends to break
        d['abs_std'] = np.std(percentage)

        # Square percentage metric
        sqr_percentage = (1 - (self.output - self.input) ** 2 / self.input) * 100
        d['sqr_mean'] = np.mean(sqr_percentage)
        d['sqr_med'] = np.median(sqr_percentage)
        d['sqr_std'] = np.std(sqr_percentage)

        # Verification results
        d['div_max'], d['div_min'], d['div_avg'] = Model.verification(self.output, print_res=False)
        self.dict_perf = d
        return d

    # END LOGIC METHODS

    # SKELETON FUNCTIONS
    # Function to be overwritten in subclasses, perform lower level operations
    def _fit_model(self, train_array: np.array, val_array: np.array or None = None) -> None:
        """
        Fits the model on the training data: skeleton, overwritten in each subclass
        val_array is optional; required by Keras for training
        :param train_array: numpy array, time series training data
        :param val_array: numpy array, optional, time series validation data
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def _get_code(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the latent space from the given input: skeleton, overwritten in each subclass
        :input_: numpy array, time series input
        :return: numpy array, time series code
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    def _get_output(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the decoded data given the latent space: skeleton, overwritten in each subclass
        :input_: numpy array, time series code
        :return: numpy array, time series output
        """
        raise NotImplementedError("Skeleton not filled by subclass")

    # END SKELETONS

    # BEGIN GENERAL METHODS
    # Static functions used for all models
    @staticmethod
    def data_reading(re: float = 40.0, nx: int = 24, nu: int = 2, shuf: bool = True) -> np.array:
        """
        Function to read H5 files with flow data, can change Re to run for different flows
        :param re: float, Reynolds Number (20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0)
        :param nx: int, size of the grid
        :param nu: int, components of the velocity vector (1, 2)
        :param shuf: boolean, if true returns shuffled data
        :return: numpy array, Time series for given flow with shape [#frames, nx, nx, nu]
        """

        # File selection
        # T has different values depending on Re
        if re == 20.0 or re == 30.0 or re == 40.0:
            T = 20000
        else:
            T = 2000

        dir_curr = os.path.split(__file__)[0]
        path_rel = ('SampleFlows', f'Kolmogorov_Re{re}_T{T}_DT01.h5')
        path = os.path.join(dir_curr, *path_rel)

        # Read dataset
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        # Instantiating the velocities array with zeros
        u_all = np.zeros((nx, nx, len(t), nu))

        # Update u_all with data from file
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
        if nu == 2:
            u_all[:, :, :, 1] = np.array(hf.get('v_refined'))

        # Transpose of u_all in order to make it easier to work with it
        # Old dimensions -> [nx, nx, frames, nu]
        # New dimensions -> [frames, nx, nx, nu]
        u_all = np.transpose(u_all, [2, 0, 1, 3])
        hf.close()

        # Shuffle of the data in order to make sure that there is heterogeneity throughout the test set
        if shuf:
            u_all = shuffle(u_all, random_state=42)

        return u_all

    @staticmethod
    def preprocess(u_all: np.array or None = None, re: float = 40.0, nx: int = 24, nu: int = 2,
                   split: bool = True, norm: bool = True) -> np.array or tuple[np.array]:
        """
        Function to preprocess the dataset. It can split into train validation and test, and normalize the values
        :param u_all: numpy array, optional, time series flow velocities
        :param re: float, Reynolds Number (20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0)
        :param nx: int, size of the grid
        :param nu: int, components of the velocity vector (1, 2)
        :param split: bool, if True the data will be divided among train (75%), validation (20%) and test (5%)
        :param norm: bool, if True the data will be normalized for values between 0 and 1
        :return: numpy array(s), depending on "split" it will return the velocity time series after processing
        """

        # Scenario where no data is provided by the user
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
            val_ratio = int(np.round(0.75 * len(u_all)))
            test_ratio = int(np.round(0.95 * len(u_all)))

            u_train = u_all[:val_ratio, :, :, :].astype('float32')
            u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
            u_test = u_all[test_ratio:, :, :, :].astype('float32')
            return u_train, u_val, u_test

        return u_all

    @staticmethod
    def train_test_batch(param_ranges: dict, model: object, save: bool = False) -> None:
        """
        Function to tune a model using different hyperparameters
        Trains, evaluates and writes results to file for a model and with hyperparameter ranges
        :param param_ranges: dict, Hyperparameters to tune as keys, with their ranges as values
        :param model: Model object, subclass model that needs to be tuned # not sure object is the correct type hint
        :param save: bool, saves the model (only implemented for AE)
        :return: None, results written to timestamped file
        """

        # Preprocess dataset
        u_train, u_val, u_test = Model.preprocess()

        # Flattened grid of all combinations
        param_grid = ParameterGrid(param_ranges)

        # Loop over model combinations
        n = 0  # Counter
        dir_ = os.path.join(os.path.split(__file__)[0], 'TuningDivision', 'Raw')
        _name = f'_at_{datetime.now().strftime("%m.%d.%Y_%Hh%Mm")}.csv'
        flag = False
        for params in param_grid:
            start_time = time.time()  # get start time

            # initialise model with parameters, specify training set for hot start
            model_ = model(**params, train_array=u_train, val_array=u_val)
            model_.u_test = u_test
            model_.passthrough(u_test)  # sets input and output

            end_time = time.time()  # get end time
            t_time = end_time-start_time
            # compute loss
            perf = model_.performance()

            # write to file
            write = {**perf}
            write.update(params)

            columns = write.keys()
            with open(os.path.join(dir_, f'{model_.__class__.__name__}{_name}')
                      , 'a', newline='') as f:
                writer = DictWriter(f, columns)

                if not flag:  # write column names, its ugly im sorry
                    labels = dict(write)
                    for key in labels.keys():
                        labels[key] = key
                    writer.writerow(labels)
                    flag = True

                writer.writerow(write)  # write results
            print(f'{model_.__class__.__name__} {n} tuned')
            n += 1

            if save:
                if model_.__class__.__name__ == 'AE':
                    dir_2 = os.path.join(os.path.split(__file__)[0], 'KerasModels', 'Raw')
                    model_.encoder.save(os.path.join(dir_2, f'encoder_s_dim={model_.dimensions[-1]}.h5'))
                    model_.decoder.save(os.path.join(dir_2, f'decoder_s_dim={model_.dimensions[-1]}.h5'))
                    model_.autoencoder.save(os.path.join(dir_2, f'autoencoder_s_dim={model_.dimensions[-1]}.h5'))
                    print(f'Saved: {model_.__class__.__name__} with dim {model_.dimensions[-1]} to {dir_2}')
                else:
                    print('Save model setting exclusive to AE')

    @staticmethod
    def verification(data: np.array, print_res: bool = True) -> tuple[float, float, float]:
        """
        Function to check conservation of mass
        :param data: numpy array, time series 2D velocity grid
        :param print_res: bool, if True results are printed
        :return: tuple of floats -> max, min, and avg of divergence of velocity
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

        max_div = np.max(all_conv)
        min_div = np.min(all_conv)
        avg_div = np.sum(np.abs(all_conv))/len(all_conv)
        if print_res:
            print(f'max: {max_div}')
            print(f'min: {min_div}')
            print(f'avg: {avg_div}')

        return max_div, min_div, avg_div

    @staticmethod
    def energy(nxnx2: np.array) -> np.array:
        """
        Function to calculate energy of a singular frame
        :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
        :return: numpy array, kinetic grid wise energy of one image without taking mass into account
        """

        u = nxnx2[:, :, 0]
        v = nxnx2[:, :, 1]
        return 0.5 * np.add(np.multiply(u, u), np.multiply(v, v))

    @staticmethod
    def curl(nxnx2: np.array) -> np.array:
        """
        Function to calculate curl of a single time frame
        :param nxnx2:  numpy array, time frame of velocities with shape [nx,nx,2]
        :return: numpy array, curl over the grid of a picture
        """
        u = nxnx2[:, :, 0]
        v = nxnx2[:, :, 1]
        return np.subtract(np.gradient(u, axis=1), np.gradient(v, axis=0))

    @staticmethod
    def plot_energy(nxnx2: np.array) -> None:
        """
        Function to plot energy/grid without mass/density
        :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
        :return: None, plots image
        """
        plt.contourf(Model.energy(nxnx2), min=0, max=1.1)
        plt.show()

    @staticmethod
    def plot_vorticity(nxnx2: np.array) -> None:
        """
        Function to plot vorticity of a time frame
        :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
        :return: None, plots image
        """
        plt.contourf(Model.curl(nxnx2), min=-2.2, max=2.2)
        plt.show()

    @staticmethod
    def plot_velocity(nxnx2: np.array) -> None:
        """
        Function to plot velocity in a vector field
        :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
        :return: None, plots image
        """

        n = np.shape(nxnx2)[0]

        x = np.arange(n)
        y = np.arange(n)
        X, Y = np.meshgrid(x, y)

        # Creating plot
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.quiver(X, Y, nxnx2[:, :, 0], nxnx2[:, :, 1])

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        plt.show()

    @staticmethod
    def u_v_plot(nxnx2: np.array, vmin=0.0, vmax=1.1, title=None, color='viridis') -> None:
        """
        Plots velocity components x, y in different plots
        :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
        :return: None, plots images
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.contourf(nxnx2[:, :, 0], vmin=vmin, vmax=vmax, cmap=color)
        ax1.title.set_text('u velocity')

        ax2 = fig.add_subplot(122)
        ax2.contourf(nxnx2[:, :, 1], vmin=vmin, vmax=vmax, cmap=color)
        ax2.title.set_text('v velocity')

        fig.suptitle(title)
        plt.show()

    @staticmethod
    def plot_all(nxnx2: np.array) -> None:
        """
        Function combining different plotting options for a single time frame
        :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
        :return: None, plots images
        """
        Model.u_v_plot(nxnx2)
        Model.plot_energy(nxnx2)
        Model.plot_vorticity(nxnx2)
        Model.plot_velocity(nxnx2)
    # END GENERAL METHODS
