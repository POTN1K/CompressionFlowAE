import numpy as np

from Main import AE

from keras.layers import Layer
from keras.layers import Input, Conv2D, UpSampling2D, Conv2DTranspose
import tensorflow as tf
from tensorflow import concat


class Filter(Layer):
    def __init__(self, n=1, **kwargs):
        super(Filter, self).__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        super(Filter, self).build(input_shape)

    def call(self, x1, x2, x3, x4):
        temp = np.shape(x1)[0]
        if temp is None:
            temp = 1
        if self.n < 4:
            x4 = [[[[0 for i in range(1)] for j in range(1)] for t in range(1)] for p in range(temp)]
        if self.n < 3:
            x3 = [[[[0 for i in range(1)] for j in range(1)] for t in range(1)] for p in range(temp)]
        if self.n < 2:
            x2 = [[[[0 for i in range(1)] for j in range(1)] for t in range(1)] for p in range(temp)]
        #print(x1,x2,x3,x4)
        x = concat([x1, x2, x3, x4], axis=3)
        return x

def hierarchicalNetwork(self, n):
    # Input
    self.image = Input(shape=(self.nx, self.nx, self.nu))

    # Beginning of Encoder 1
    x1 = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
    x1 = self.pooling_function((2, 2), padding='same')(x1)
    x1 = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x1)
    x1 = self.pooling_function((2, 2), padding='same')(x1)
    x1 = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x1)
    x1 = self.pooling_function((2, 2), padding='same')(x1)
    x1 = Conv2D(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x1)
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
    x2 = Conv2D(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x2)
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
    x3 = Conv2D(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x3)
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
    x4 = Conv2D(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x4)
    encoded_4 = self.pooling_function((3, 3), padding='same')(x4)
    self.encoder4 = tf.keras.models.Model(self.image, encoded_4)
    self.encoder4.trainable = False

    # Combine in filter
    self.latent_filtered = Filter(n)(encoded_1, encoded_2, encoded_3, encoded_4)

    # Decoder
    x = Conv2DTranspose(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(
        self.latent_filtered)
    x = UpSampling2D((3, 3))(x)
    x = Conv2DTranspose(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(x)
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


setattr(AE, 'h_network', hierarchicalNetwork)

# --------------------------------------------------------------------------------------------------
# Preprocess Data
train, val, test = AE.preprocess(nu=2)
train = train[:40]
val = val[:40]
test = test[:40]

AE.u_v_plot(test[0])

model = AE(dimensions=[64, 32, 16, 1], l_rate=0.0005, epochs=50, batch=20)

# Train 1st component
model.h_network(1)
model.encoder1.trainable = True
model.fit(train, val)
model.encoder1.trainable = False
t1 = model.passthrough(test[0])
#perf = model.performance()
#print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
#print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t1)


# Recompile model
w1 = model.autoencoder.get_weights()
model.h_network(2)
model.autoencoder.compile()
model.autoencoder.set_weights(w1)

# Train 2nd component
model.encoder2.trainable = True
model.fit(train, val)
model.encoder2.trainable = False
t2 = model.passthrough(test[0])
#perf = model.performance()
#print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
#print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t2)
print(model.encode((test[0])))

# Recompile model
w2 = model.autoencoder.get_weights()
model.h_network(3)
model.autoencoder.compile()
model.autoencoder.set_weights(w2)

# Train 3rd component
model.n = 3
model.encoder3.trainable = True
model.fit(train, val)
model.encoder3.trainable = False
t3 = model.passthrough(test[0])
#perf = model.performance()
#print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
#print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t3)
print(model.encode((test[0])))

# Recompile model
w3 = model.autoencoder.get_weights()
model.h_network(4)
model.autoencoder.compile()
model.autoencoder.set_weights(w3)

# Train 4th component
model.latent_filtered.n = 4
model.encoder4.trainable = True
model.fit(train, val)
model.encoder4.trainable = False
t4 = model.passthrough(test)
perf = model.performance()
print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t4[0])
print(model.encode((test[0])))
