import h5py
import numpy as np

def read(Re=20.0):
    # FILE SELECTION
    # choose between Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0

    # T has different values depending on Re
    if Re == 20.0 or Re == 30.0 or Re == 40.0:
        T = 20000
    else:
        T = 2000

    path_folder = 'SampleFlows/'  # path to folder in which flow data is situated
    path = path_folder + f'Kolmogorov_Re{Re}_T{T}_DT01.h5'

    # READ DATASET
    hf = h5py.File(path, 'r')
    Nx = 24
    Nu = 1
    t = np.array(hf.get('t'))
    u_all = np.zeros((Nx, Nx, len(t), Nu))
    u_all[:, :, :, 0] = np.array(hf.get('u_refined'))

    u_all = np.transpose(u_all, [2, 0, 1, 3])
    hf.close()
    print(u_all.shape)
    print('Read Dataset')

    # normalize data
    u_min = np.amin(u_all[:, :, :, 0])
    u_max = np.amax(u_all[:, :, :, 0])
    u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
    if Nu == 2:
        u_all[:, :, :, 1] = np.array(hf.get('v_refined'))
        v_min = np.amin(u_all[:, :, :, 1])
        v_max = np.amax(u_all[:, :, :, 1])
        u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)
    print('Normalized Data')
    return Nx, Nu, u_all
