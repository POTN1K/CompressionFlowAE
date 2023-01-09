from Main import Model, AE, POD
import numpy as np

def run_model_AE():
    data = AE.preprocess(split=False)

def run_model_POD():
    u_train, u_val, u_test = POD.preprocess(nu=2)
    train_pod = np.concatenate((u_train, u_val, u_test))
    model = POD(train_pod)
    u_test = np.concatenate((u_train, u_val, u_test))

    print(u_test-POD.preprocess(nu=2, split=False))

    model.passthrough(POD.preprocess(nu=2, split=False))
    model.verification(model.input)
    model.verification(model.output)

    model.passthrough(u_test)
    model.verification(model.input)
    model.verification(model.output)

def run_tune():
    param_ranges_dict = {'l_rate': [0.0005],
                         'epochs': [500],
                         'batch': [10],
                         'early_stopping': [10],
                         'dimensions': [
                             [504, 252, 126, 63],
                             [512, 256, 128, 64]
                         ]
                         }

    Model.train_test_batch(param_ranges_dict, AE)


if __name__ == '__main__':
    run_model_POD()

#
# import os
#
# dir_curr = os.getcwd()
# path_rel = ('Main', 'SampleFlows')
#
# path = os.path.join(dir_curr, *path_rel)
#
# print(os.listdir(path))