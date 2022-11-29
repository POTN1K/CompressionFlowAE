# Libraries
import h5py
import numpy as np


# Generic Model
class Model:
    @staticmethod
    def data_reading(re, nx, nu):
        """Function to read the H5 files, can change Re to run for different flows
            Re- Reynolds Number
            Nu- Dimension of Velocity Vector
            Nx- Size of grid"""
        # FILE SELECTION
        # choose between Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0

        # T has different values depending on Re
        if re == 20.0 or re == 30.0 or re == 40.0:
            T = 20000
        else:
            T = 2000

        path_folder = '../SampleFlows/'  # path to folder in which flow data is situated
        path = path_folder + f'Kolmogorov_Re{re}_T{T}_DT01.h5'

        # READ DATASET
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        u_all = np.zeros((nx, nx, len(t), nu))
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))  # Update u_all with data from file
        if nu == 2:
            u_all[:, :, :, 1] = np.array(hf.get('v_refined'))
        u_all = np.transpose(u_all, [2, 0, 1, 3])  # Time, Nx, Nx, Nu
        hf.close()
        return u_all

    @staticmethod
    def preprocess(u_all=None, re=20.0, nx=24, nu=1):
        if u_all is None:
            u_all = Model.data_reading(re, nx, nu)

        # normalize data
        u_min = np.amin(u_all[:, :, :, 0])
        u_max = np.amax(u_all[:, :, :, 0])
        u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
        if nu == 2:
            v_min = np.amin(u_all[:, :, :, 1])
            v_max = np.amax(u_all[:, :, :, 1])
            u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)

        val_ratio = int(np.round(0.75 * len(u_all)))  # Amount of data used for validation
        test_ratio = int(np.round(0.95 * len(u_all)))  # Amount of data used for testing

        u_train = u_all[:val_ratio, :, :, :].astype('float32')
        u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
        u_test = u_all[test_ratio:, :, :, :].astype('float32')
        return u_train, u_val, u_test
