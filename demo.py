from Main import Model, AE, POD

# def run_model_AE():
#     data = AE.preprocess(split=False)
#     model = AE(data)
#
# def run_model_POD():
#     data = POD.preprocess(split=False)
#     model = POD(data)
#     print(model.code)
#     model.plot_contributions()


def run_parent():
    data = AE.preprocess()
    print(data[0])


if __name__ == '__main__':
    run_parent()

#
# import os
#
# dir_curr = os.getcwd()
# path_rel = ('Main', 'SampleFlows')
#
# path = os.path.join(dir_curr, *path_rel)
#
# print(os.listdir(path))