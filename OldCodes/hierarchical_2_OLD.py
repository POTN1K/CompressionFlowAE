import numpy as np

from Main import AE

from keras.layers import Layer
from keras.layers import Input, Conv2D, UpSampling2D, Conv2DTranspose
import tensorflow as tf
from tensorflow import concat


# Custom Layer
class RefactorEncoded(Layer):
    def __init__(self, n_components=0, model=None, **kwargs):
        super(RefactorEncoded, self).__init__(**kwargs)
        self.n_components = n_components
        self.model = model

    def build(self, input_shape):
        super(RefactorEncoded, self).build(input_shape)

    def call(self, x, image=None):
        if self.n_components != 0:
            value = self.model.encoder(image)
            x = concat([value, x], 3)
        return x


def hierarchicalNetwork(self, n=0, model=None):
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

    # Custom layer
    complete_encoded = RefactorEncoded(n, model)(encoded, self.image)
    print(f'Complete encoded: {complete_encoded}')
    # Beginning of Decoder
    x = Conv2DTranspose(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(
        complete_encoded)
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
    self.encoder = tf.keras.models.Model(self.image, complete_encoded)

    # Creation of decoder
    encoded_input = Input(shape=(1, 1, complete_encoded.shape[3]))  # latent vector definition
    deco = self.autoencoder.layers[-9](encoded_input)  # re-use the same layers as the ones of the autoencoder
    for i in range(8):
        deco = self.autoencoder.layers[-8 + i](deco)
    self.decoder = tf.keras.models.Model(encoded_input, deco)


setattr(AE, 'h_network', hierarchicalNetwork)

# --------------------------------------------------------------------------------------------------
# Preprocess Data
train, val, test = AE.preprocess(nu=2)
#train = train[:50]
#val = val[:50]
#test = test[:50]

AE.u_v_plot(test[0])

# Create 1 component autoencoder
print("Model 1")
model1 = AE(dimensions=[64, 32, 16, 1], l_rate=0.0005, epochs=50, batch=20)
model1.h_network()
model1.fit(train, val)
t1 = model1.passthrough(test)
perf = model1.performance()

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

AE.u_v_plot(t1[0])
#print(model1.encoder.get_weights())
model1.encoder.trainable = False
# ___________________________________________________________________________________________
print("Model 2")
# Create 2 component
model2 = AE(dimensions=[64, 32, 10, 1], l_rate=0.0005, epochs=75, batch=20)
model2.h_network(1, model1)
model2.fit(train, val)
t2 = model2.passthrough(test)
perf = model2.performance()

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

AE.u_v_plot(t2[0])
#print(model1.encoder.get_weights())
print(model2.encode(test[0]))
model2.encoder.trainable = False
## ___________________________________________________________________________________________
print("Model 3")
# Create 3 component
model3 = AE(dimensions=[64, 32, 10, 1], l_rate=0.0005, epochs=100, batch=20)
model3.h_network(2, model2)
model3.fit(train, val)
t3 = model3.passthrough(test)
perf = model3.performance()
#
print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
#
AE.u_v_plot(t3[0])
print(model3.encode((test[0])))
model2.encoder.trainable = False

print("Model 4")
# Create 3 component
model4 = AE(dimensions=[64, 32, 10, 1], l_rate=0.0005, epochs=125, batch=20)
model4.h_network(3, model3)
model4.fit(train, val)
t4 = model4.passthrough(test)
perf = model4.performance()
#
print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
#
AE.u_v_plot(t4[0])
print(model4.encode((test[0])))
