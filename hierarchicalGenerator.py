""" Hierarchical Autoencoder Generator

This file creates a hierarchical autoencoder in a script style.
It creates a 4 component latent space, where the components are organized by importance.

The code does the following:
    1. Preprocess the data and initializes an AE object
    2. Sets the filter to train only using the necessary components
    3. Unlocks the weights of the encoder to be trained
    4. Trains encoder and decoder
    5. Locks the trained weights of the encoder
    6. Saves the weights and recompiles the model with a modified filter*
    7. Loads the previously trained weights
    8. Repeat steps 3 to 7 until completely trained

* The filter is set in h_network(n), where n is the number of components to be used.
For training it must be started with 1, since the principal component needs to be trained.
The filter will block all output from the other three encoders, since they are not yet necessary.
After one encoder is done, a new encoder should be unlocked.
By doing this, the second encoder will be trained while still considering what the first encoder can reproduce.
In this way, encoder2 will build on top of encoder1. Always remember to lock the weights of the already trained
encoders to avoid overwriting what was already trained.

The model presented here has an absolute accuracy of 91.76%, and a squared accuracy of 99.3%
This model has already been trained and can be called using AE.create_trained()
"""

# Local Libraries
from FlowCompression import AE, custom_loss_function

# Preprocess Data
train, val, test = AE.preprocess()

# Show image to be compared
print("Original flow")
AE.u_v_plot(test[0])

# Instantiation of model
model = AE(dimensions=[32, 16, 8, 4], l_rate=0.0005, epochs=50, batch=20)
# model.loss = custom_loss_function -> Uncomment if want to train using other loss (not recommended, slow)

# Train 1st component
print("First component")
model.h_network(1)
model.encoder1.trainable = True
model.fit(train, val)
model.encoder1.trainable = False
t1 = model.passthrough(test)
perf = model.performance()
print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t1[0])
print(model.encode((test[0])))

# Recompile model
w1 = model.autoencoder.get_weights()
model.h_network(2)
model.autoencoder.compile()
model.autoencoder.set_weights(w1)

# Train 2nd component
print("Second component")
model.encoder2.trainable = True
model.fit(train, val)
model.encoder2.trainable = False
t2 = model.passthrough(test)
perf = model.performance()
print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t2[0])
print(model.encode((test[0])))

# Recompile model
w2 = model.autoencoder.get_weights()
model.h_network(3)
model.autoencoder.compile()
model.autoencoder.set_weights(w2)

# Train 3rd component
print("Third component")
model.encoder3.trainable = True
model.fit(train, val)
model.encoder3.trainable = False
t3 = model.passthrough(test)
perf = model.performance()
print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
AE.u_v_plot(t3[0])
print(model.encode((test[0])))

# Recompile model
w3 = model.autoencoder.get_weights()
model.h_network(4)
model.autoencoder.compile()
model.autoencoder.set_weights(w3)

# Train 4th component
print("Fourth component")
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

# Divergence verification
model.verification(model.y_pred)

# Uncomment to save model
#model.autoencoder.trainable = True
#model.autoencoder.save('autoencoder_h.h5')
#model.encoder.save('encoder_h.h5')
#model.decoder.save('decoder_h.h5')