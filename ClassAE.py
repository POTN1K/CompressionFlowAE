# Libraries
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, AveragePooling2D, UpSampling2D, concatenate,
    BatchNormalization, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Model Class
class Model:
    def __init__(self, dimensions=[8, 4, 2, 1], activation_function='tanh', l_rate=0.01, epochs=10, batch=250,
                 early_stopping=5, pooling='max', re=20.0, Nu=1, Nx=24, loss='mse'):
        self.dimensions = dimensions
        self.activation_function = activation_function
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch = batch
        self.early_stopping = early_stopping
        self.re = re
        self.nu = Nu
        self.nx = Nx
        self.pooling = pooling
        self.loss = loss
        # Instantiating
        self.u_all = None
        self.u_train = None
        self.u_val = None
        self.u_test = None
        self.hist = None
        self.image = None
        self.encoded = None
        self.decoded = None

    @property
    def pooling(self):
        return self._pooling

    @pooling.setter
    def pooling(self, value):
        if value == 'max':
            self.pooling_function = MaxPool2D
        else:
            self.pooling_function = AveragePooling2D

    def data_reading(self):
        """Function to read the H5 files, can change Re to run for different flows
            Re- Reynolds Number
            Nu- Dimension of Velocity Vector
            Nx- Size of grid"""
        # FILE SELECTION
        # choose between Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0

        # T has different values depending on Re
        if self.re == 20.0 or self.re == 30.0 or self.re == 40.0:
            T = 20000
        else:
            T = 2000

        path_folder = 'SampleFlows/'  # path to folder in which flow data is situated
        path = path_folder + f'Kolmogorov_Re{self.re}_T{T}_DT01.h5'

        # READ DATASET
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        u_all = np.zeros((self.nx, self.nx, len(t), self.nu))
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))  # Update u_all with data from file
        if self.nu == 2:
            u_all[:, :, :, 1] = np.array(hf.get('v_refined'))
        u_all = np.transpose(u_all, [2, 0, 1, 3])  # Time, Nx, Nx, Nu
        hf.close()
        self.u_all = u_all
        print(f'Shape of initial u dataset: {u_all.shape}')
        print('Read Dataset')

    def preprocess(self):
        if self.u_all is None:
            self.data_reading()

        # normalize data
        u_min = np.amin(self.u_all[:, :, :, 0])
        u_max = np.amax(self.u_all[:, :, :, 0])
        self.u_all[:, :, :, 0] = (self.u_all[:, :, :, 0] - u_min) / (u_max - u_min)
        if self.nu == 2:
            v_min = np.amin(self.u_all[:, :, :, 1])
            v_max = np.amax(self.u_all[:, :, :, 1])
            self.u_all[:, :, :, 1] = (self.u_all[:, :, :, 1] - v_min) / (v_max - v_min)
        print('Normalized Data')

        val_ratio = int(np.round(0.75 * len(self.u_all)))  # Amount of data used for validation
        test_ratio = int(np.round(0.95 * len(self.u_all)))  # Amount of data used for testing

        self.u_train = self.u_all[:val_ratio, :, :, :].astype('float32')
        self.u_val = self.u_all[val_ratio:test_ratio, :, :, :].astype('float32')
        self.u_test = self.u_all[test_ratio:, :, :, :].astype('float32')

    def input_image(self):
        self.image = Input(shape=(self.nx, self.nx, self.nu))

    def network(self):
        x = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
        x = self.pooling_function((2, 2), padding='same')(x)
        x = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
        x = self.pooling_function((2, 2), padding='same')(x)
        x = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
        x = self.pooling_function((2, 2), padding='same')(x)
        x = Conv2D(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x)
        self.encoded = self.pooling_function((3, 3), padding='same')(x)
        x = Conv2DTranspose(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(
            self.encoded)
        x = UpSampling2D((3, 3))(x)
        x = Conv2DTranspose(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(x)
        x = UpSampling2D((2, 2))(x)
        self.decoded = Conv2DTranspose(self.nu, (3, 3), activation='linear', padding='same')(x)

    def creator(self):
        self.autoencoder = tf.keras.models.Model(self.image, self.decoded)
        # self.encoder = tf.keras.models.Model(self.image, self.encoded)
        encoded_input = Input(shape=(1, 1, self.encoded.shape[3]))  # encoded_input == latent vector
        deco = self.autoencoder.layers[-7](encoded_input)  # we re-use the same layers as the ones of the autoencoder
        for i in range(6):
            deco = self.autoencoder.layers[-6 + i](deco)

        self.decoder = tf.keras.models.Model(encoded_input, deco)

    def training(self):
        # Create checkpoints
        self.autoencoder.compile(optimizer=Adam(learning_rate=self.l_rate), loss=self.loss)
        checkpoint_filepath = './checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stopping)

        # Create the history of the model
        self.hist = self.autoencoder.fit(self.u_train, self.u_train, epochs=self.epochs, batch_size=self.batch,
                                         shuffle=True, validation_data=(self.u_val, self.u_val),
                                         verbose=1,
                                         callbacks=[model_checkpoint_callback, early_stop_callback])

    def visual_analysis(self):
        y_nn = self.autoencoder.predict(self.u_test[0:1, :, :, :])

        # u velocity
        plt.subplot(121)
        figure = plt.contourf(y_nn[0, :, :, 0], vmin=0.0, vmax=1.1)
        plt.subplot(122)
        figure2 = plt.contourf(self.u_test[0, :, :, 0], vmin=0.0, vmax=1.1)
        plt.colorbar(figure2)
        plt.title("Velocity x-dir")
        plt.show()

        # v velocity
        if self.nu == 2:
            plt.subplot(121)
            figure = plt.contourf(y_nn[0, :, :, 1], vmin=0.0, vmax=1.1)
            plt.subplot(122)
            figure2 = plt.contourf(self.u_test[0, :, :, 1], vmin=0.0, vmax=1.1)
            plt.colorbar(figure2)
            # fig = plt.figure()
            # ax = fig.add_subplot(121)
            # ax.contourf(y_nn[0, :, :, 1], vmin=0.0, vmax=1.5)
            # ax = fig.add_subplot(122)
            # ax.contourf(self.u_test[0, :, :, 1], vmin=0.0, vmax=1.5)
            plt.title("Velocity y-dir")
            plt.show()

        loss_history = self.hist.history['loss']
        val_history = self.hist.history['val_loss']
        plt.plot(loss_history, 'b', label='Training loss')
        plt.plot(val_history, 'r', label='Validation loss')
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = Model()
    model.data_reading()
    model.preprocess()
    model.input_image()
    model.network()
    model.creator()
    model.training()
    model.visual_analysis()
