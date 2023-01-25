from Main import Model, AE, POD, custom_loss_function
import numpy as np


def run_model_AE():
    data = AE.preprocess(nu=2)
    data_cartesian = AE.preprocess(nu=2)
    # convert
    for set_cartesian in data_cartesian:
        # tx 24, 24, 2 -> 24, 24, 2
        length = np.sqrt(set_cartesian[:, :, :, 0] ** 2 + set_cartesian[:, :, :, 1] ** 2)
        phase = np.sin(set_cartesian[:, :, :, 0] / set_cartesian[:, :, :, 0])
        set_radial = np.concatenate((length, phase))
        print(set_radial.shape)


def run_model_POD():
    data = POD.preprocess(split=False)
    model = POD(data)
    print(model.code)
    model.plot_contributions()


def run_tune():
    dim = None
    # dim = [[16, 8, 4, 2], [24, 12, 6, 3]]       # comment to generate simple structure from latent space dim range

    if dim is None:
        dim = []
        for latent_dim in range(24, 65, 2):
            dim.append([8 * latent_dim, 4 * latent_dim, 2 * latent_dim, latent_dim])

    param_ranges_dict = {'l_rate': [0.0005],
                         'epochs': [200],
                         'batch': [10],
                         'early_stopping': [10],
                         'dimensions': dim}

    Model.train_test_batch(param_ranges_dict, AE, save=True)


def tune_physical():
    n = 2
    u_train, u_val, u_test = AE.preprocess(nu=n)

    model = AE.create_trained(2)
    model.u_train, model.u_val, model.u_test = u_train, u_val, u_test
    model.loss, model.l_rate, model.epochs = custom_loss_function, 0.0000000001, 20

    print('Original')
    model.verification(u_test)

    print('Model divergence')
    model.passthrough(u_test)
    perf = model.performance()
    model.verification(model.y_pred)
    print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
    print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

    print('Tuned divergence')
    model.fit(u_train, u_val)
    model.passthrough(u_test)
    perf = model.performance()
    model.verification(model.y_pred)
    print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
    print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')

    # model.autoencoder.save('autoencoder_p.h5')
    # model.encoder.save('encoder_p.h5')
    # model.decoder.save('decoder_p.h5')


if __name__ == '__main__':
    run_tune()

#
# import os
#
# dir_curr = os.getcwd()
# path_rel = ('Main', 'SampleFlows')
#
# path = os.path.join(dir_curr, *path_rel)
#
# print(os.listdir(path))
