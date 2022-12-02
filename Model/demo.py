from Model.Models.ClassAE import AE
from Model.Models.ClassPOD import POD

def run_model_AE():
    data = AE.preprocess(split=False)
    model = AE(data)

def run_model_POD():
    data = POD.preprocess(split=False)
    model = POD(data)
    print(model.code)
    model.plot_contributions()


if __name__ == '__main__':
    run_model_POD()

