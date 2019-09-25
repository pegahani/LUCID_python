import pickle
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

"upload real curve"
with open('real_data.pickle', 'rb') as handle:
    df_matrix = pickle.load(handle)
real_y_test = df_matrix[:, 0]

class normal_dtw:
    def __init__(self, ncsimul, test):
        self.real = real_y_test
        if ncsimul.shape[1] is not None:
            self.ncsimul = ncsimul[:,0]
        else:
            self.ncsimul = ncsimul
        self.test = test

        self.real_x_test = [i for i in range(len(self.real))]
        self.ncsimul_x_test = [i for i in range(len(ncsimul))]

        self.real_size = len(self.real)
        ncsimul_size = len(ncsimul)
        self.step = round(self.real_size / ncsimul_size)

    "x normalisation"
    def average_x(self):
        "average X of the curves"
        self.new_real = []
        counter = 0
        while counter * self.step < self.real_size:
            self.new_real.append(np.mean(self.real[counter * self.step:(counter + 1) * self.step]))
            counter += 1

        self.new_real = np.asarray(self.new_real)
        self.new_real_x = [i for i in range(len(self.new_real))]

        return

    "y normalization"
    def __normalize(self, data):
        # Store the data's original shape
        shape = data.shape
        # Flatten the data to 1 dimension
        data = np.reshape(data, (-1,))
        # Find minimum and maximum
        maximum = np.max(data)
        minimum = np.min(data)
        # Create a new array for storing normalized values
        normalized_values = list()
        # Iterate through every value in data
        for x in data:
            # Normalize
            x_normalized = (x - minimum) / (maximum - minimum)
            # Append it in the array
            normalized_values.append(x_normalized)
        # Convert to numpy array
        n_array = np.array(normalized_values)
        # Reshape the array to its original shape and return it.
        return np.array(np.reshape(n_array, shape), dtype='float64')

    def total_normalisation(self):
        self.ncsimul_y_normal = self.__normalize(self.ncsimul)
        self.new_real_normal = self.__normalize(self.new_real)
        return

    def dtw_(self, length_min, length_max):
        path = dtw.warping_path(self.new_real_normal[length_min:length_max], self.ncsimul_y_normal[length_min:length_max])
        distance, paths = dtw.warping_paths(self.new_real_normal[length_min:length_max],
                                            self.ncsimul_y_normal[length_min:length_max])

        dtwvis.plot_warping(self.new_real_normal[length_min:length_max], self.ncsimul_y_normal[length_min:length_max], path,
                            filename="warp"+str(self.test)+".png")

        best_path = dtw.best_path(paths)
        dtwvis.plot_warpingpaths(self.new_real_normal[length_min:length_max], self.ncsimul_y_normal[length_min:length_max], paths,
                                 best_path, filename="best_path"+str(self.test)+".png")


"""
print("execute")
adr_1 = '../../Data/energy_data/phillipe/AcierCourt_Seq1.xlsx'
adr_2 = '../../Data/energy_data/phillipe/AcierCourt_Seq2.xlsx'
adr_3 = '../../Data/energy_data/phillipe/AcierCourt_Seq3.xlsx'
adr_4 = '../../Data/energy_data/phillipe/AcierCourt_Seq4.xlsx'


df_power_1= pandas.read_excel(adr_1)
df_power_2= pandas.read_excel(adr_2)
df_power_3= pandas.read_excel(adr_3)
df_power_4= pandas.read_excel(adr_4)


df_matrix_1 = df_power_1.values
df_matrix_2 = df_power_2.values
df_matrix_3 = df_power_3.values
df_matrix_4 = df_power_4.values

args = (df_matrix_1, df_matrix_2, df_matrix_3, df_matrix_4)
df_matrix = np.concatenate(args)
#print(df_matrix.shape)
"""
#
# with open('real_data.pickle', 'wb') as handle:
#     pickle.dump(df_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # "upload ncsimul curve"
# with open('ncsimul_data.pickle', 'rb') as handle:
#     ncsimul_y_test = pickle.load(handle)
#
# print(real_y_test.shape)
# exp = normal_dtw(real_y_test, ncsimul_y_test)
# exp.average_x()
# exp.total_normalisation()



