from Main import AE

# --------------------------------------------------------------------------------------------------
# Preprocess Data
train, val, test = AE.preprocess(nu=2)
#train = train[:40]
#val = val[:40]
#test = test[:40]

print("Original flow")
AE.u_v_plot(test[0])

model = AE(dimensions=[32, 16, 8, 4], l_rate=0.0005, epochs=100, batch=20)

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
model.n = 3
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

model.autoencoder.trainable = True
model.autoencoder.save('autoencoder_h.h5')
model.encoder.save('encoder_h.h5')
model.decoder.save('decoder_h.h5')

# Accuracy
# Absolute - 91.76%
# Squared - 99.3%