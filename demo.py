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
    param_ranges_dict = {'l_rate': [0.0005],
                         'epochs': [500],
                         'batch': [10],
                         'early_stopping': [10],
                         'dimensions': [
                         [176, 88, 44, 22],
                         [192, 96, 48, 24],
                         [208, 104, 52, 26],
                         [224, 112, 56, 28],
                         [240, 120, 60, 30]
                         ]
                         }

    Model.train_test_batch(param_ranges_dict, AE)


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