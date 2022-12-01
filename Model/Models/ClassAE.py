# Libraries
from Model.ParentClass import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Autoencoder Model Class
class AE(Model):
    def __init__(self, dimensions=[8, 4, 2, 1], activation_function='tanh', l_rate=0.01, epochs=10, batch=200,
                 early_stopping=5, pooling='max', re=40.0, Nu=1, Nx=24, loss='mse'):
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
                                         verbose=0,
                                         callbacks=[model_checkpoint_callback, early_stop_callback])

    def visual_analysis(self):
        y_nn = self.autoencoder.predict(self.u_test[0:1, :, :, :], verbose=0)

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

    def performance(self):
        self.mse = self.autoencoder.evaluate(self.u_test, self.u_test, self.batch, verbose=0)


def run_model():
    u_train, u_val, u_test = AE.preprocess()

    model = AE()
    model.u_train = u_train
    model.u_val = u_val
    model.u_test = u_test
    model.input_image()
    model.network()
    model.creator()
    model.training()
    model.visual_analysis()
    model.performance()

    print(model.autoencoder.summary())


if __name__ == '__main__':
    run_model()
