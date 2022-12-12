from Main import Model, AE, POD

def run_model_AE():
    data = AE.preprocess(split=False)

#
# def run_model_POD():
#     data = POD.preprocess(split=False)
#     model = POD(data)
#     print(model.code)
#     model.plot_contributions()


def run_tune():
    param_ranges_dict = {'l_rate': [0.01, 0.001],
                         'epochs': [10, 50, 100],
                         'batch': [1000, 100, 10],
                         'early_stopping': [5, 10, 20],
                         'dimensions': [[8, 4, 2, 1], [16, 8, 4, 2], [24, 12, 6, 3]]}

    Model.train_test_batch(param_ranges_dict, AE)


if __name__ == '__main__':
    run_model_AE()

#
# import os
#
# dir_curr = os.getcwd()
# path_rel = ('Main', 'SampleFlows')
#
# path = os.path.join(dir_curr, *path_rel)
#
# print(os.listdir(path))