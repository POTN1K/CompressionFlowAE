# Code that defines the Autoencoder object. It has n principal methods:
#               - network: creates the architecture of the NN
#               - creator: which creates the encoder, decoder and autoencoder attributes
#               - training: to train the autoencoder (it automatically trains the decoder and encoder too)
#               - visual analysis: plots the first 10 reconstructed flows and the error evolution throught the epochs
#               - performance: calculates mse of every image and transforms it into a percentage of accuracy


# Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.models import load_model
import os

# Local codes
import sys

sys.path.append('.')
from Main import Model
from Main.ExperimentsAE import Filter, custom_loss_function


# Uncomment if keras does not run
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Autoencoder Model Class
class AE(Model):

    @staticmethod
    def create_trained(h=True):
        # Load Models
        dir_curr = os.path.split(__file__)[0]
        if h:
            model = AE(dimensions=[32, 16, 8, 4], l_rate=0.0005, epochs=100, batch=20)
            model.h_network(4)
            # Autoencoder
            auto_rel = ('KerasModels', f'autoencoder_h.h5')
            model.autoencoder = load_model(os.path.join(dir_curr, *auto_rel), custom_objects={'Filter': Filter})
            # Encoder
            enco_rel = ('KerasModels', f'encoder_h.h5')
            model.encoder = load_model(os.path.join(dir_curr, *enco_rel), custom_objects={'Filter': Filter})
            # Decoder
            deco_rel = ('KerasModels', f'decoder_h.h5')
            model.decoder = load_model(os.path.join(dir_curr, *deco_rel), custom_objects={'Filter': Filter})
        else:
            model = AE()
            # Autoencoder
            auto_rel = ('KerasModels', f'autoencoder_2D.h5')
            model.autoencoder = load_model(os.path.join(dir_curr, *auto_rel))
            # Encoder
            enco_rel = ('KerasModels', f'encoder_2D.h5')
            model.encoder = load_model(os.path.join(dir_curr, *enco_rel))
            # Decoder
            deco_rel = ('KerasModels', f'decoder_2D.h5')
            model.decoder = load_model(os.path.join(dir_curr, *deco_rel))
        model.trained = True
        return model

    # Initialize model
    def __init__(self, dimensions=[32, 16, 8, 4], activation_function='tanh', l_rate=0.0005, epochs=500, batch=10,
                 early_stopping=10, pooling='max', re=40.0, nu=2, nx=24, loss='mse', train_array=None, val_array=None,
                 trained=False):
        """ Ambiguous Inputs-
            dimensions: Number of features per convolution layer, dimensions[-1] is dimension of latent space.
            pooling: 'max' or 'ave', function to combine pixels.
            nu: Number of velocity components, 1 -> 'x', 2 -> 'x','y'.
            nx: Size of grid side."""
        self.dimensions = dimensions
        self.activation_function = activation_function
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch = batch
        self.early_stopping = early_stopping
        self.re = re
        self.nu = nu
        self.nx = nx
        self.pooling = pooling
        self.loss = loss
        self.trained = trained
        # Instantiating
        self.u_all = None
        self.u_train = None
        self.u_val = None
        self.u_test = None
        self.hist = None
        self.image = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.hist = None
        self.y_pred = None

        self.network()
        super().__init__(train_array=train_array, val_array=val_array)

    @property
    def pooling(self):
        return self.pooling_function

    @pooling.setter
    def pooling(self, value):
        if value == 'max':
            self.pooling_function = MaxPool2D
        else:
            self.pooling_function = AveragePooling2D

    # SKELETON FUNCTIONS
    def fit_model(self, train_array: np.array, val_array: np.array or None = None) -> None:  # skeleton
        """
        Fits the model on the training data: skeleton, overwrite
        val_array is optional; required by Keras for training
        :param train_array: time series training data
        :param val_array: optional, time series validation data
        """
        self.u_train = train_array
        self.u_val = val_array

        self.training()

    def get_code(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the encoded signal given data to encode
        :input_: time series input
        :return: time series code
        """
        self.u_test = input_
        return self.encoder.predict(input_, verbose=0)

    def get_output(self, input_: np.array) -> np.array:  # skeleton
        """
        Returns the decoded data given the encoded signal
        :input_: time series code
        :return: time series output
        """
        self.y_pred = self.decoder.predict(input_, verbose = 0)
        return self.y_pred

    # END SKELETONS

    def network(self):
        """ This function defines the architecture of the neural network. Every layer of the NN presents a convolution
            and a pooling. There are 8 layers in total, 4 for the encoder and 4 for the decoder.
            Connv2D inputs:
            - number of features to be extracted from the frame
            - (m, n) defines the size of the filter, x_size_filter = x_size_frame / m, y_size_filter = y_size_frame / n
            - how to behave when at the boundary of the frame, 'same' adds 0s to the left/right/up/down of the frame
            pooling_function inputs:
            - size of the area of frame where the pooling is performed
            - padding (check Conv2D)
            """
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

        """ Use previously defined architecture to build an untrained model.
            It will create an autoencoder, encoder, and decoder. All three trained together."""
        # Creation of autoencoder
        self.autoencoder = tf.keras.models.Model(self.image, decoded)

        # Creation of enconder
        self.encoder = tf.keras.models.Model(self.image, encoded)

        # Creation of decoder
        encoded_input = Input(shape=(1, 1, encoded.shape[3]))  # latent vector definition
        deco = self.autoencoder.layers[-9](encoded_input)  # re-use the same layers as the ones of the autoencoder
        for i in range(8):
            deco = self.autoencoder.layers[-8 + i](deco)
        self.decoder = tf.keras.models.Model(encoded_input, deco)

    def h_network(self, n):
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

        # Complete model
        self.autoencoder = tf.keras.models.Model(self.image, decoded)
        self.encoder = tf.keras.models.Model(self.image, self.latent_filtered)

        encoded_input = Input(shape=(1, 1, self.latent_filtered.shape[3]))  # latent vector definition
        deco = self.autoencoder.layers[-9](encoded_input)  # re-use the same layers as the ones of the autoencoder
        for i in range(8):
            deco = self.autoencoder.layers[-8 + i](deco)
        self.decoder = tf.keras.models.Model(encoded_input, deco)

    def training(self):
        """Function to train autoencoder, with checkpoints for checkpoints and early stopping"""

        self.autoencoder.compile(optimizer=Adam(learning_rate=self.l_rate), loss=self.loss)

        # Early stop callback creation
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping)

        # Fit the training and validation data to model, while saving a history. Verbose prints the epochs
        self.hist = self.autoencoder.fit(self.u_train, self.u_train, epochs=self.epochs, batch_size=self.batch,
                                         shuffle=False, validation_data=(self.u_val, self.u_val),
                                         verbose=1,
                                         callbacks=[early_stop_callback])

        # Predict model using test data
        # self.y_pred = self.autoencoder.predict(self.u_test[:, :, :, :], verbose=0)

    def visual_analysis(self, n=1, plot_error=False):
        """Function to visualize some samples of predictions in order to visually compare with the test set. Moreover,
            the evolution of the error throughout the epochs is plotted"""

        for i in range(n):

            # Set of predictions we are going to plot. We decided on the first 10 frames but it could be whatever
            image_to_plot = self.y_pred[i:i + 1, :, :, :]

            # u velocity

            min_ = np.min(self.u_test[i, :, :, 0])
            max_ = np.max(self.u_test[i, :, :, 0])

            plt.subplot(121)
            figure1x = plt.contourf(self.y_pred[i, :, :, 0], vmin=min_, vmax=max_)
            plt.subplot(122)
            figure2x = plt.contourf(self.u_test[i, :, :, 0], vmin=min_, vmax=max_)
            plt.colorbar(figure2x)
            plt.title("Velocity x-dir")
            plt.show()

            # v velocity
            min_ = np.min(self.u_test[i, :, :, 1])
            max_ = np.max(self.u_test[i, :, :, 1])
            if self.nu == 2:
                plt.subplot(121)
                figure1y = plt.contourf(self.y_pred[i, :, :, 1], vmin=min_, vmax=max_)
                plt.subplot(122)
                figure2y = plt.contourf(self.u_test[i, :, :, 1], vmin=min_, vmax=max_)
                plt.colorbar(figure2y)
                plt.title("Velocity y-dir")
                plt.show()

        if plot_error:
            # Creation of a loss graph, comparing validation and training data.
            loss_history = self.hist.history['loss']
            val_history = self.hist.history['val_loss']
            plt.plot(loss_history, 'b', label='Training loss')
            plt.plot(val_history, 'r', label='Validation loss')
            plt.title("Loss History")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def vorticity_energy(self):
        """
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.contourf(AE.curl(self.y_pred[0, :, :, 0]))
        ax1.set_title('predicted')
        ax2.contourf(AE.curl(self.u_test[0, :, :, 0]))
        ax1.set_title('true')
        plt.show()
        """
        min_ = np.min(AE.curl(self.u_test[0, :, :, :]))
        max_ = np.max(AE.curl(self.u_test[0, :, :, :]))
        plt.subplot(121)
        plt.contourf(AE.curl(self.y_pred[0, :, :, :]), vmin=min_, vmax=max_)
        plt.subplot(122)
        plt.contourf(AE.curl(self.u_test[0, :, :, :]), vmin=min_, vmax=max_)
        plt.title("Vorticity")
        plt.show()

        min_ = np.min(AE.energy(self.u_test[0, :, :, :]))
        max_ = np.max(AE.energy(self.u_test[0, :, :, :]))
        plt.subplot(121)
        plt.contourf(AE.energy(self.y_pred[0, :, :, :]), vmin=min_, vmax=max_)
        plt.subplot(122)
        plt.contourf(AE.energy(self.u_test[0, :, :, :]), vmin=min_, vmax=max_)
        plt.title("Energy")
        plt.show()
        return None

    def performance(self):
        """Here we transform the mse into an accuracy value. Two different metrics are used, the absolute
        error and the squared error. With those values, two different stds are calculated"""

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


def run_model():
    """General function to run one model"""

    n = 2
    u_train, u_val, u_test = AE.preprocess(nu=n)

    model = AE.create_trained(False)
    # model = AE()

    model.u_train, model.u_val, model.u_test = u_train, u_val, u_test

    # model.fit(u_train, u_val)

    model.passthrough(u_test)
    model.visual_analysis()
    model.vorticity_energy()
    perf = model.performance()

    model.verification(u_test)
    model.verification(model.y_pred)

    print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
    print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')


if __name__ == '__main__':
    run_model()
