""" Autoencoder Model class

This file defines a class "AE", which is a subclass of Model. It creates a customizable Keras model,
with different architectures and hyperparameters.
"""

# Libraries
# Utils
import os
import sys

sys.path.append('.')
# Numerics
import numpy as np
# Machine Learning
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.models import load_model
# Plotting
import matplotlib.pyplot as plt
# Local Library
from Main import Model, Filter, custom_loss_function


# Uncomment if keras does not run
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Autoencoder Model Class
class AE(Model):
    """
    Autoencoder class, subclass from Model. Creates an autoencoder object using a keras model.

    Main attributes:
    - Model hyperparameters: dimensions, activation_function, l_rate, epochs, batch, early_stopping, pooling, loss
    - Data: u_train, u_val, u_test
    - Keras models: autoencoder, encoder, decoder
    - Other: hierarchical, re, nx, nu, y_pred

    Parent methods:
    - fit: trains model on training and validation data
    - encode: returns latent space
    - decode: returns decoded latent space
    - passthrough: encodes and decodes flow
    - performance: measures performance

    Main methods:
    - create_trained: static, loads a trained model
    - network/h_network: initializes model architecture
    - training: trains model
    - plot_loss_history: plots the loss history after training
    - vorticity_energy: plots energy and vorticity comparison
    """

    # Initialize model
    def __init__(self, dimensions: list[int] = [32, 16, 8, 4], l_rate: float = 0.0005, epochs: int = 200,
                 batch: int = 10, early_stopping: int = 10, re: float = 40.0, nx: int = 24, nu: int = 2,
                 activation_function: str = 'tanh', pooling: str = 'max', loss: str = 'mse',
                 train_array: np.array or None = None, val_array: np.array or None = None,
                 hierarchical: bool = False) -> None:
        """
        Initialize AE object
        :param dimensions: list, sets the dimensionality of output for each convolution layer.
                            dimensions[3] sets the size of the latent space
        :param l_rate: float, learning rate
        :param epochs: int, number of epochs
        :param batch: int, number of batches
        :param early_stopping: int, number of similar elements before stop training
        :param re: float, Reynolds number
        :param nx: int, size of the grid
        :param nu: int, components of velocity vector (1,2)
        :param activation_function: str
        :param pooling: str, pooling technique ('max','ave')
        :param loss: str, loss function
        :param train_array: numpy array, optional, training data
        :param val_array: numpy array, optional, validation data
        :param hierarchical: bool, sets the model to be a hierarchical autoencoder or general
        """

        self.dimensions = dimensions
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch = batch
        self.early_stopping = early_stopping
        self.re = re
        self.nx = nx
        self.nu = nu
        self.activation_function = activation_function
        self.pooling = pooling
        self.loss = loss
        self.hierarchical = hierarchical

        # Instantiating
        self.u_train, self.u_val, self.u_test = None, None, None
        self.autoencoder, self.encoder, self.decoder = None, None, None
        self.image = None
        self.hist = None
        self.y_pred = None

        if self.hierarchical:
            self.h_network()
        else:
            self.network()

        super().__init__(train_array=train_array, val_array=val_array)

    # BEGIN PROPERTIES
    @property
    def pooling(self):
        """
        Return pooling function
        :return: str, pooling function
        """
        return self.pooling_function

    @pooling.setter
    def pooling(self, value):
        """
        Set pooling function
        :param value: str, pooling function name ('max', 'ave')
        :return: str, pooling function
        """
        if value == 'max':
            self.pooling_function = MaxPool2D
        elif value == 'ave':
            self.pooling_function = AveragePooling2D
        else:
            raise ValueError("Use a valid pooling function")

    # END PROPERTIES

    # SKELETON FUNCTIONS
    # Overwriting of low level codes for parent class
    def _fit_model(self, train_array: np.array, val_array: np.array = None) -> None:
        """
        Fits the model on the training data: skeleton, overwrite
        val_array is optional; required by Keras for training
        :param train_array: numpy array, time series training data
        :param val_array: numpy array, time series validation data
        :return: None
        """

        self.u_train = train_array
        self.u_val = val_array

        self.training()

    def _get_code(self, input_: np.array) -> np.array:
        """
        Returns the latent space from the given input
        :param input_: numpy array, time series input
        :return: numpy array, time series latent space
        """

        self.u_test = input_
        return self.encoder.predict(input_, verbose=0)

    def _get_output(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the decoded latent space
        :param input_: numpy array, latent space
        :return: numpy array, time series
        """

        self.y_pred = self.decoder.predict(input_, verbose=0)
        return self.y_pred

    # END SKELETONS

    # BEGIN MODEL FUNCTIONS
    # Network creation methods
    def network(self) -> None:
        """
        This function defines the architecture of the neural network. Every layer of the NN presents a convolution
        and a pooling. There are 8 layers in total, 4 for the encoder and 4 for the decoder.
            Conv2D inputs:
            - number of features to be extracted from the frame
            - (m, n) defines the size of the filter, x_size_filter = x_size_frame / m, y_size_filter = y_size_frame / n
            - how to behave when at the boundary of the frame, 'same' adds 0s to the left/right/up/down of the frame
            pooling_function inputs:
            - size of the area of the frame where the pooling is performed
            - padding (check Conv2D)
        :return: None
        """
        # Input
        self.image = Input(shape=(self.nx, self.nx, self.nu))

        # Beginning of Autoencoder and Encoder
        x = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
        x = self.pooling_function((2, 2), padding='same')(x)
        x = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
        x = self.pooling_function((2, 2), padding='same')(x)
        x = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
        x = self.pooling_function((2, 2), padding='same')(x)
        x = Conv2D(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x)
        encoded = self.pooling_function((3, 3), padding='same')(x)
        # End of Encoder

        # Beginning of Decoder
        x = Conv2DTranspose(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(encoded)
        x = UpSampling2D((3, 3))(x)
        x = Conv2DTranspose(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2DTranspose(self.nu, (3, 3), activation='linear', padding='same')(x)
        # End of Decoder and Autoencoder

        # Use the architecture to define an autoencoder, encoder, decoder
        # Creation of autoencoder
        self.autoencoder = tf.keras.models.Model(self.image, decoded)
        # Creation of encoder
        self.encoder = tf.keras.models.Model(self.image, encoded)
        # Creation of decoder
        encoded_input = Input(shape=(1, 1, encoded.shape[3]))  # latent vector definition
        deco = self.autoencoder.layers[-9](encoded_input)  # re-use the same layers as the ones of the autoencoder
        for i in range(8):
            deco = self.autoencoder.layers[-8 + i](deco)
        self.decoder = tf.keras.models.Model(encoded_input, deco)

    def h_network(self, n=4) -> None:
        """
        Creation of a 4 size latent space autoencoder. It has 4 different encoders,
        so each mode is trained separately, ranked by importance. They share one decoder.
        To train each encoder, the filter needs to be adapted, the weights locked, and the model recompiled.
        Refer to hierarchicalGenerator.py for more information
        :param n: Number of components allowed by the filter (1-4)
        :return: None
        """
        # Input
        self.image = Input(shape=(self.nx, self.nx, self.nu))

        # Beginning of Encoder 1
        x1 = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
        x1 = self.pooling_function((2, 2), padding='same')(x1)
        x1 = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x1)
        x1 = self.pooling_function((2, 2), padding='same')(x1)
        x1 = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x1)
        x1 = self.pooling_function((2, 2), padding='same')(x1)
        x1 = Conv2D(1, (3, 3), padding='same', activation=self.activation_function)(x1)
        encoded_1 = self.pooling_function((3, 3), padding='same')(x1)
        self.encoder1 = tf.keras.models.Model(self.image, encoded_1)
        self.encoder1.trainable = False

        # Beginning of Encoder 2
        x2 = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
        x2 = self.pooling_function((2, 2), padding='same')(x2)
        x2 = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x2)
        x2 = self.pooling_function((2, 2), padding='same')(x2)
        x2 = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x2)
        x2 = self.pooling_function((2, 2), padding='same')(x2)
        x2 = Conv2D(1, (3, 3), padding='same', activation=self.activation_function)(x2)
        encoded_2 = self.pooling_function((3, 3), padding='same')(x2)
        self.encoder2 = tf.keras.models.Model(self.image, encoded_2)
        self.encoder2.trainable = False

        # Beginning of Encoder 3
        x3 = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
        x3 = self.pooling_function((2, 2), padding='same')(x3)
        x3 = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x3)
        x3 = self.pooling_function((2, 2), padding='same')(x3)
        x3 = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x3)
        x3 = self.pooling_function((2, 2), padding='same')(x3)
        x3 = Conv2D(1, (3, 3), padding='same', activation=self.activation_function)(x3)
        encoded_3 = self.pooling_function((3, 3), padding='same')(x3)
        self.encoder3 = tf.keras.models.Model(self.image, encoded_3)
        self.encoder3.trainable = False

        # Beginning of Encoder 4
        x4 = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
        x4 = self.pooling_function((2, 2), padding='same')(x4)
        x4 = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x4)
        x4 = self.pooling_function((2, 2), padding='same')(x4)
        x4 = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x4)
        x4 = self.pooling_function((2, 2), padding='same')(x4)
        x4 = Conv2D(1, (3, 3), padding='same', activation=self.activation_function)(x4)
        encoded_4 = self.pooling_function((3, 3), padding='same')(x4)
        self.encoder4 = tf.keras.models.Model(self.image, encoded_4)
        self.encoder4.trainable = False

        # Combine in filter
        self.latent_filtered = Filter(n)(encoded_1, encoded_2, encoded_3, encoded_4)

        # Decoder
        x = Conv2DTranspose(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(
            self.latent_filtered)
        x = UpSampling2D((3, 3))(x)
        x = Conv2DTranspose(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2DTranspose(self.nu, (3, 3), activation='linear', padding='same')(x)

        # Use the architecture to define an autoencoder, encoder, decoder
        # Creation of autoencoder
        self.autoencoder = tf.keras.models.Model(self.image, decoded)
        # Creation of encoder
        self.encoder = tf.keras.models.Model(self.image, self.latent_filtered)
        # Creation of decoder
        encoded_input = Input(shape=(1, 1, self.latent_filtered.shape[3]))  # latent vector definition
        deco = self.autoencoder.layers[-9](encoded_input)  # re-use the same layers as the ones of the autoencoder
        for i in range(8):
            deco = self.autoencoder.layers[-8 + i](deco)
        self.decoder = tf.keras.models.Model(encoded_input, deco)

    def training(self) -> None:
        """
        Function to train autoencoder, with early stopping
        :return: None
        """

        # Compile of model
        self.autoencoder.compile(optimizer=Adam(learning_rate=self.l_rate), loss=self.loss)

        # Early stop callback creation
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping)

        # Fit the training and validation data to model, while saving a history. Verbose prints the epochs
        self.hist = self.autoencoder.fit(self.u_train, self.u_train, epochs=self.epochs, batch_size=self.batch,
                                         shuffle=False, validation_data=(self.u_val, self.u_val),
                                         verbose=1,
                                         callbacks=[early_stop_callback])

    def plot_loss_history(self) -> None:
        """
        Plot of a loss graph, comparing validation and training data.
        :return: None, plot of loss history
        """

        if self.hist is None:
            raise ValueError("The model needs to be trained first")

        loss_history = self.hist.history['loss']
        val_history = self.hist.history['val_loss']
        plt.plot(loss_history, 'b', label='Training loss')
        plt.plot(val_history, 'r', label='Validation loss')
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def vorticity_energy(self) -> None:
        """
        Plot of vorticity and energy comparison between original and reconstructed flow
        :return: None, Plot images
        """

        # Curl plot
        min_ = np.min(AE.curl(self.u_test[0, :, :, :]))
        max_ = np.max(AE.curl(self.u_test[0, :, :, :]))
        plt.subplot(121)
        plt.contourf(AE.curl(self.y_pred[0, :, :, :]), vmin=min_, vmax=max_)
        plt.subplot(122)
        plt.contourf(AE.curl(self.u_test[0, :, :, :]), vmin=min_, vmax=max_)
        plt.title("Vorticity")
        plt.show()

        # Energy plot
        min_ = np.min(AE.energy(self.u_test[0, :, :, :]))
        max_ = np.max(AE.energy(self.u_test[0, :, :, :]))
        plt.subplot(121)
        plt.contourf(AE.energy(self.y_pred[0, :, :, :]), vmin=min_, vmax=max_)
        plt.subplot(122)
        plt.contourf(AE.energy(self.u_test[0, :, :, :]), vmin=min_, vmax=max_)
        plt.title("Energy")
        plt.show()

    def performance(self) -> dict[str, float]:
        """
        Function to create dictionary with various performance metrics.
        Keys - 'mse', 'abs_percentage', 'abs_std', 'sqr_percentage', 'sqr_std'
        :return: dict, performance metrics
        """

        d = dict()
        # Calculation of MSE
        d['mse'] = self.autoencoder.evaluate(self.u_test, self.u_test, self.batch, verbose=0)

        # Absolute percentage metric, along with its std
        d['abs_percentage'] = np.average(1 - np.abs(self.y_pred - self.u_test) / self.u_test) * 100
        abs_average_images = np.average((1 - np.abs(self.y_pred - self.u_test) / self.u_test), axis=(1, 2)) * 100
        d['abs_std'] = np.std(abs_average_images)

        # Squared percentage metric, along with std
        d['sqr_percentage'] = np.average(1 - (self.y_pred - self.u_test) ** 2 / self.u_test) * 100
        sqr_average_images = np.average((1 - (self.y_pred - self.u_test) ** 2 / self.u_test), axis=(1, 2)) * 100
        d['sqr_std'] = np.std(sqr_average_images)
        self.dict_perf = d
        return d

    @staticmethod
    def create_trained(type: int = 1) -> object:
        """
        Function to load a trained model. It can load a hierarchical or standard model.
        :param type: int, selects trained model (1-Physical, 2-Hierarchical, 3-TunedIntermediate)
        :return: AE object, trained model object
        """
        # Load Models
        dir_curr = os.path.split(__file__)[0]
        match type:
            case 1:
                model = AE(hierarchical=False)
                # Autoencoder
                auto_rel = ('KerasModels', f'autoencoder_p.h5')
                model.autoencoder = load_model(os.path.join(dir_curr, *auto_rel), custom_objects={'custom_loss_function': custom_loss_function})
                # Encoder
                enco_rel = ('KerasModels', f'encoder_p.h5')
                model.encoder = load_model(os.path.join(dir_curr, *enco_rel), custom_objects={'custom_loss_function': custom_loss_function})
                # Decoder
                deco_rel = ('KerasModels', f'decoder_p.h5')
                model.decoder = load_model(os.path.join(dir_curr, *deco_rel), custom_objects={'custom_loss_function': custom_loss_function})
            case 2:
                model = AE(hierarchical=True)
                # Autoencoder
                auto_rel = ('KerasModels', f'autoencoder_h.h5')
                model.autoencoder = load_model(os.path.join(dir_curr, *auto_rel), custom_objects={'Filter': Filter})
                # Encoder
                enco_rel = ('KerasModels', f'encoder_h.h5')
                model.encoder = load_model(os.path.join(dir_curr, *enco_rel), custom_objects={'Filter': Filter})
                # Decoder
                deco_rel = ('KerasModels', f'decoder_h.h5')
                model.decoder = load_model(os.path.join(dir_curr, *deco_rel), custom_objects={'Filter': Filter})
            case 3:
                model = AE(hierarchical=False)
                # Autoencoder
                auto_rel = ('KerasModels', f'autoencoder_2D.h5')
                model.autoencoder = load_model(os.path.join(dir_curr, *auto_rel))
                # Encoder
                enco_rel = ('KerasModels', f'encoder_2D.h5')
                model.encoder = load_model(os.path.join(dir_curr, *enco_rel))
                # Decoder
                deco_rel = ('KerasModels', f'decoder_2D.h5')
                model.decoder = load_model(os.path.join(dir_curr, *deco_rel))
            case _:
                raise ValueError("Give a valid number for model")
        model.trained = True
        return model


if __name__ == '__main__':
    u_train, u_val, u_test = AE.preprocess()

    model = AE.create_trained(1)  # -> Comment model = AE(), and model.fit() to run pre trained
    # model = AE()

    model.u_train, model.u_val, model.u_test = u_train, u_val, u_test

    # model.fit(u_train, u_val)

    t = model.passthrough(u_test)
    # model.vorticity_energy()
    perf = model.performance()
    AE.plot_all(t[0])

    model.verification(u_test)
    model.verification(model.y_pred)

    print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
    print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
