import numpy as np

from FlowCompression import AE

from keras.layers import Layer
from keras.layers import Input, Conv2D, UpSampling2D, Conv2DTranspose
import tensorflow as tf
from tensorflow import concat
from tqdm import tqdm


# Custom Layer
class RefactorEncoded(Layer):
    def __init__(self, n_components=0, pred=dict(), **kwargs):
        super(RefactorEncoded, self).__init__(**kwargs)
        self.n_components = n_components
        self.predictions = pred

    def build(self, input_shape):
        super(RefactorEncoded, self).build(input_shape)

    #   def compute_output_shape(self, input_shape):
    #      return (None, 1, 1, input_shape[3] + self.n_components)

    def call(self, x, image=None):
        if self.n_components == 0:
            return x
        # Call previous model on image and combine with x
        value = self.predictions.get(image.numpy().tobytes())
        print(value)
        if value is not None:
            x = concat([value, x], 3)
            print(x)
        return x



def hierarchicalNetwork(self, n=0, pred=None):
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
    complete_encoded = RefactorEncoded(n, pred)(encoded, self.image)

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
train = train[:10]
val = val[:10]
test = test[:10]

#AE.plot_all(test[0])

# Create 1 component autoencoder
print("Model 1")
model1 = AE(dimensions=[8, 4, 2, 1], l_rate=0.001, epochs=30, batch=20)
model1.h_network()
model1.fit(train, val)
t1 = model1.passthrough(test)
perf = model1.performance()

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

#AE.plot_all(t1[0])

# Create dictionary
data = np.concatenate((train, val, test))
pred = dict()
for image in tqdm(data):
    pred[image.tobytes()] = model1.encode(image)

print(pred[data[0].tobytes()])
# ___________________________________________________________________________________________
print("Model 2")
# Create 2 component
model2 = AE(dimensions=[8, 4, 2, 1], l_rate=0.001, epochs=30, batch=20)
model2.h_network(1, pred)
model2.fit(train, val)
t2 = model2.passthrough(test)
perf = model2.performance()

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

#AE.plot_all(t2[0])

## Create dictionary
#pred2 = dict()
#for image in tqdm(data):
#    pred2[image.tobytes()] = model2.encode(image)
#
#print(pred2[data[0].tobytes()])
## ___________________________________________________________________________________________
#print("Model 3")
## Create 3 component
#model3 = AE(dimensions=[8, 4, 2, 1], l_rate=0.001, epochs=30, batch=20)
#model3.h_network(2, pred2)
#model3.fit(train, val)
#t3 = model3.passthrough(test)
#perf = model3.performance()
#
#print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
#print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
#
#AE.plot_all(t3[0])
#print(model3.encode((data[0])))