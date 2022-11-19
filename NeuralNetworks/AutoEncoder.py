# Import Libraries
import numpy as np
import tensorflow as tf
# from datetime import datetime
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, AveragePooling2D, UpSampling2D, concatenate, \
    BatchNormalization, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# -------------------------------------------------------------------------------------------------
# FILE SELECTION
Re = 20.0  # choose between Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0

# T has different values depending on Re
if Re == 20.0 or Re == 30.0 or Re == 40.0:
    T = 20000
else:
    T = 2000

path_folder = 'Final_Product/'  # path to folder in which flow data is situated
path = path_folder + f'Kolmogorov_Re{Re}_T{T}_DT01.h5'

# -------------------------------------------------------------------------------------------------
# PARAMETER SELECTION

dimensions = [8, 4, 2, 1]  # Specify the dimensions for each encoding layer (dim[3] = dimension of latent space)
# e.g. [16,8,4,2], [32,16,8,4], [64,32,16,8], [128,64,32,16], [256, 128, 64, 32]
# Decoding layer is mirror of encoding layer

activation_function = 'tanh'  # specify which activation function to use
learning_rate = 0.1  # change to 0.0001 if using a latent space dimension of 32 and 64
number_epochs = 10  # set high, early stopping function will stop iterations if it sees no visible change after 'patience_early_stopping' number of epochs
patience_early_stopping = 20  # how many epochs with no significant change before AE stops training
pooling = 'max'  # set to 'max' or 'average'
if pooling == 'max':
    pooling_function = MaxPool2D
else:
    pooling_function = AveragePooling2D

# -------------------------------------------------------------------------------------------------
# READ DATASET
hf = h5py.File(path, 'r')
Nx = 24
Nu = 1
t = np.array(hf.get('t'))
u_all = np.zeros((Nx, Nx, len(t), Nu))
u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
# u_all[:,:,:,1] = np.array(hf.get('v_refined'))
u_all = np.transpose(u_all, [2, 0, 1, 3])
hf.close()
print(u_all.shape)
print('Read Dataset')

# normalize data
u_min = np.amin(u_all[:, :, :, 0])
u_max = np.amax(u_all[:, :, :, 0])
u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
if Nu == 2:
    v_min = np.amin(u_all[:, :, :, 1])
    v_max = np.amax(u_all[:, :, :, 1])
    u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)
print('Normalized Data')

# visualization of the dataset
fig = plt.figure()
ax = fig.add_subplot(121)
ax.contourf(u_all[50, :, :, 0])
# ax2 = fig.add_subplot(122)
# ax2.contourf(u_all[0,:,:,1])
plt.show()
print('Showed data')

# -------------------------------------------------------------------------------------------------
# PREPARE DATASET
val_ratio = int(np.round(0.75 * len(u_all)))
test_ratio = int(np.round(0.95 * len(u_all)))

u_train = u_all[:val_ratio, :, :, :].astype('float32')
u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
u_test = u_all[test_ratio:, :, :, :].astype('float32')
print('Prepared Dataset')

# -------------------------------------------------------------------------------------------------
# CREATING NETWORK
input_img = Input(shape=(Nx, Nx, Nu))

nb_layer = 0

x = Conv2D(dimensions[0], (3, 3), padding='same', activation=activation_function)(input_img)
x = pooling_function((2, 2), padding='same')(x)
x = Conv2D(dimensions[1], (3, 3), padding='same', activation=activation_function)(x)
x = pooling_function((2, 2), padding='same')(x)
x = Conv2D(dimensions[2], (3, 3), padding='same', activation=activation_function)(x)
x = pooling_function((2, 2), padding='same')(x)
x = Conv2D(dimensions[3], (3, 3), padding='same', activation=activation_function)(x)
encoded = pooling_function((3, 3), padding='same')(x)

x = Conv2DTranspose(dimensions[3], (3, 3), padding='same', activation=activation_function)(encoded)
x = UpSampling2D((3, 3))(x)
x = Conv2DTranspose(dimensions[2], (3, 3), padding='same', activation=activation_function)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(dimensions[1], (3, 3), padding='same', activation=activation_function)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(dimensions[0], (3, 3), padding='same', activation=activation_function)(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(Nu, (3, 3), activation='linear', padding='same')(x)

autoencoder = tf.keras.models.Model(input_img, decoded)
encoder = tf.keras.models.Model(input_img, encoded)
print('Encoder Created')

# definition of the decoder
encoded_input = Input(shape=(1, 1, encoded.shape[3]))
deco = autoencoder.layers[-7](encoded_input)  # we re-use the same layers as the ones of the autoencoder
for i in range(6):
    deco = autoencoder.layers[-6 + i](deco)

decoder = tf.keras.models.Model(encoded_input, deco)

print('Decoder Created')
print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

# -------------------------------------------------------------------------------------------------
# TRAINING NETWORK
autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
checkpoint_filepath = './checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early_stopping)

nb_epochs = number_epochs
batch_size = 20
hist = autoencoder.fit(u_train, u_train, epochs=nb_epochs, batch_size=batch_size,
                       shuffle=True, validation_data=(u_test, u_test),
                       verbose=1,
                       callbacks=[model_checkpoint_callback, early_stop_callback])

loss_history = hist.history['loss']
val_history = hist.history['val_loss']
plt.plot(loss_history, 'b')
plt.plot(val_history, 'r')
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()
print('Trained Model')
# -------------------------------------------------------------------------------------------------
# VISUALIZATION OF PREDICTION
y_nn = autoencoder.predict(u_test[0:1, :, :, :])
print(y_nn.shape)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.contourf(y_nn[0, :, :, 0])
ax = fig.add_subplot(122)
ax.contourf(u_test[0, :, :, 0])
plt.show()

print(f'AE error: {loss_history[-1]}')
