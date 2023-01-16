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
# from .ExperimentsAE.CustomLibraries import MixedPooling2D


# Uncomment if keras does not run
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# loss function
def central_difference(before, after):
    return tf.math.scalar_mul(0.5, tf.math.subtract(after, before))

def forward_difference(current, after):
    return tf.math.subtract(after, current)

def backward_difference(before, current):
    return tf.math.subtract(current, before)

def one_gradient(kxnxn, i, j, axis):
    if axis == 0:
        if i == 0:
            return forward_difference(kxnxn[:,i+1,j], kxnxn[:,i,j])
        elif i == 23:
            return backward_difference(kxnxn[:,i,j], kxnxn[:,i-1,j])
        else:
            return central_difference(kxnxn[:,i+1,j], kxnxn[:,i-1,j])
    elif axis == 1:
        if j == 0:
            return forward_difference(kxnxn[:,i,j+1], kxnxn[:,i,j])
        elif j == 23:
            return backward_difference(kxnxn[:,i,j], kxnxn[:,i,j-1])
        else:
            return central_difference(kxnxn[:,i,j+1], kxnxn[:,i,j-1])
    else:
        raise NotImplementedError

@tf.function
def custom_gradient(kxnxn, axis):
    n = 24
    return tf.transpose([[one_gradient(kxnxn, i, j, axis = axis) for i in range(n)] for j in range(n)], (2, 0, 1))

def custom_loss_function(y_true, y_pred):
            u_true = y_true[:,:,:,0]
            v_true = y_true[:,:,:,1]
            u_pred = y_pred[:,:,:,0]
            v_pred = y_pred[:,:,:,1]

            energy_true =   tf.math.add(tf.multiply(u_true, u_true), (tf.multiply(v_true, v_true)))
            energy_pred =   tf.math.add(tf.multiply(u_pred, u_pred), (tf.multiply(v_pred, v_pred)))

            energy_difference = tf.math.reduce_mean(tf.math.abs(tf.subtract(energy_true, energy_pred)), axis=[1,2])

            curl_true = tf.math.subtract(custom_gradient(u_true, axis = 1), custom_gradient(v_true, axis = 0))
            curl_pred = tf.math.subtract(custom_gradient(u_pred, axis = 1), custom_gradient(v_pred, axis = 0))

            curl_difference = tf.math.reduce_mean(tf.math.abs(tf.subtract(curl_true, curl_pred)), axis=[1,2])

            divergence = tf.math.abs(tf.math.reduce_mean(tf.math.add(custom_gradient(u_pred, axis = 0), custom_gradient(v_pred, axis = 1)), axis=[1,2]))

            u_diff = tf.math.subtract(u_true, u_pred)
            v_diff = tf.math.subtract(v_true, v_pred)
            u_mse =  tf.math.reduce_mean(tf.math.multiply(u_diff, u_diff), axis = [1,2])
            v_mse =  tf.math.reduce_mean(tf.math.multiply(v_diff, v_diff), axis = [1,2])

            return energy_difference, curl_difference


# Autoencoder Model Class
class AE(Model):
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

    @staticmethod
    def create_trained():
        # Load Models
        dir_curr = os.path.split(__file__)[0]
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

    # SKELETON FUNCTIONS: FILL (OVERWRITE) IN SUBCLASS
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
        self.y_pred = self.decoder.predict(input_)
        return self.y_pred

    # END SKELETONS

    @property
    def pooling(self):
        return self.pooling_function

    @pooling.setter
    def pooling(self, value):
        if value == 'max':
            self.pooling_function = MaxPool2D
        elif value == 'mix':
            self.pooling_function = MixedPooling2D
        else:
            self.pooling_function = AveragePooling2D

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
            figure1x = plt.contourf(self.y_pred[i, :, :, 0], vmin= min_, vmax=max_)
            plt.subplot(122)
            figure2x = plt.contourf(self.u_test[i, :, :, 0], vmin= min_, vmax=max_)
            plt.colorbar(figure2x)
            plt.title("Velocity x-dir")
            plt.show()

            # v velocity
                # v velocity
            min_ = np.min(self.u_test[i, :, :, 1])
            max_ = np.max(self.u_test[i, :, :, 1])
            if self.nu == 2:
                plt.subplot(121)
                figure1y = plt.contourf(self.y_pred[i, :, :, 1], vmin= min_, vmax=max_ )
                plt.subplot(122)
                figure2y = plt.contourf(self.u_test[i, :, :, 1], vmin= min_, vmax=max_ )
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

        '''
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.contourf(AE.curl(self.y_pred[0, :, :, 0]))
        ax1.set_title('predicted')
        ax2.contourf(AE.curl(self.u_test[0, :, :, 0]))
        ax1.set_title('true')
        plt.show()
        '''
        min_ = np.min(AE.curl(self.u_test[0, :, :, :]))
        max_ = np.max(AE.curl(self.u_test[0, :, :, :]))
        plt.subplot(121)
        plt.contourf(AE.curl(self.y_pred[0, :, :, :]), vmin= min_, vmax=max_)
        plt.subplot(122)
        plt.contourf(AE.curl(self.u_test[0, :, :, :]), vmin= min_, vmax=max_)
        plt.title("Vorticity")
        plt.show()

        min_ = np.min(AE.energy(self.u_test[0, :, :, :]))
        max_ = np.max(AE.energy(self.u_test[0, :, :, :]))
        plt.subplot(121)
        plt.contourf(AE.energy(self.y_pred[0, :, :, :]), vmin= min_, vmax=max_)
        plt.subplot(122)
        plt.contourf(AE.energy(self.u_test[0, :, :, :]), vmin= min_, vmax=max_)
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

    @staticmethod
    def energy(nxnx2: np.array):
        '''
        returns the kinetic grid wise energy of one image without taking mass into account 
        '''
        u = nxnx2[0]
        v = nxnx2[1]
        return 0.5 * np.add(np.multiply(u, u), np.multiply(v, v))

    @staticmethod
    def curl(nxnx2: np.array):
        '''
        returns the curl over the grid of a picture -> curl is used to calculate lift/drag therefore significant
        '''
        u = nxnx2[0]
        v = nxnx2[1]

        return np.subtract(np.gradient(u, axis = 1), np.gradient(v, axis=0))

    @staticmethod
    def plot_energy(nxnx2 : np.array):
        '''
        plots energy/grid without mass/density
        '''
        plt.contourf(AE.energy(nxnx2), min = 0, max = 1.1)
        plt.title('Energy')
        plt.show()
        return None


    @staticmethod
    def plot_vorticity(nxnx2 : np.array):
        '''
        This method returns and shows a plot of the cross product of the velocity components
        '''
        plt.contourf(AE.curl(nxnx2),  min = 0, max = 2.2)
        plt.title('Vorticity')
        plt.show()
        return None

    @staticmethod
    def plot_velocity(nxnx2: np.array):
        '''
        plots vectorfield
        '''
        x, y = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, 2 * np.pi, 24))
        plt.quiver(x, y, nxnx2[:, :, 0], nxnx2[:, :, 1])
        plt.title('Velocity')
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
        AE.u_v_plot(nxnx2)
        AE.plot_energy(nxnx2)
        AE.plot_vorticity(nxnx2)
        AE.plot_velocity(nxnx2)

def run_model():
    """General function to run one model"""

    n = 2
    u_train, u_val, u_test = AE.preprocess(nu=n)


    # model = AE.create_trained()
    model = AE()
    model.network()
    model.u_train, model.u_val, model.u_test = u_train, u_val, u_test
    model.epochs = 20
    model.l_rate = 0.1
    model.batch = 50
    # model = AE()
    model.fit(u_train, u_val)

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
