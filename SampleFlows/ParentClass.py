# Libraries
import h5py
import numpy as np
from sklearn.utils import shuffle

# Generic Model
class Model:
    @staticmethod
    def data_reading(Re, Nx, Nu):
        """Function to read the H5 files, can change Re to run for different flows
            Re- Reynolds Number
            Nu- Dimension of Velocity Vector
            Nx- Size of grid

            Final dimensions of output: [Time (number of frames), Nx, Nx, Nu]"""
        # File selection
        # Re= 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0
        # T has different values depending on Re
        if Re == 20.0 or Re == 30.0 or Re == 40.0:
            T = 20000
        else:
            T = 2000

        path_folder = '../SampleFlows/'  # path to folder in which flow data is situated
        path = path_folder + f'Kolmogorov_Re{Re}_T{T}_DT01.h5'

        # READ DATASET
        hf = h5py.File(path, 'r')
        t = np.array(hf.get('t'))
        # Instantiating the velocities array with zeros
        u_all = np.zeros((Nx, Nx, len(t), Nu))

        # Update u_all with data from file
        u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
        if Nu == 2:
            u_all[:, :, :, 1] = np.array(hf.get('v_refined'))

        # Transpose of u_all in order to make it easier to work with it
        # New dimensions of u_all = [Time, Nx, Nx, Nu]
        #       - Time: number of frames we have in our data set, which are always related to a different time moment
        #       - Nx: size of the frame in the horizontal component
        #       - Nx: size of the frame in the vertical component
        #       - Nu: dimension of the velocity vector
        u_all = np.transpose(u_all, [2, 0, 1, 3])
        hf.close()

        # Shuffle of the data in order to make sure that there is heterogeneity throughout the test set
        u_all = shuffle(u_all, random_state=42)
        return u_all

    

    @staticmethod
    def preprocess(u_all=None, Re=40.0, Nx=24, Nu=1):
        """ Function to scale the data set and split it into train, validation and test sets.
            nx: Size of the grid side
            nu: Number of velocity components, 1-> 'x', 2 -> 'x','y'"""

        # Run data reading to avoid errors
        if u_all is None:
            u_all = Model.data_reading(Re, Nx, Nu)

        # Normalize data
        u_min = np.amin(u_all[:, :, :, 0])
        u_max = np.amax(u_all[:, :, :, 0])
        u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
        if Nu == 2:
            # Code to run if using velocities in 'y' direction as well
            v_min = np.amin(u_all[:, :, :, 1])
            v_max = np.amax(u_all[:, :, :, 1])
            u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)

        # Division of training, validation and testing data
        val_ratio = int(np.round(0.75 * len(u_all)))  # Amount of data used for validation
        test_ratio = int(np.round(0.95 * len(u_all)))  # Amount of data used for testing

        u_train = u_all[:val_ratio, :, :, :].astype('float32')
        u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
        u_test = u_all[test_ratio:, :, :, :].astype('float32')
        return u_train, u_val, u_test
