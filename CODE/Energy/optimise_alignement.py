
"""
tested ncsimul: ACIER_S_1_ncsimul_F4042R.T22.025.Z03.10.csv
tested real:     ACIER_S_1_real_F4042R.T22.025.Z03.10.csv"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

adr = "../../Data/energy_data/sac_data/GP2R/"
ncsimul_file = "ACIER_S_1_ncsimul_F4042R.T22.025.Z03.10.csv"
real_file = "ACIER_S_1_real_F4042R.T22.025.Z03.10.csv"

class normalise:
    "x normalisation"
    def __init__(self):
        return

    "x normalization"
    def average_x(self, input_curve, step):
        "average X of the curves"
        new_real = []
        new_index = []
        counter = 0
        while counter * step < len(input_curve):
            new_real.append(np.mean(input_curve[counter * step:(counter + 1) * step]))
            new_index.append([counter * step, (counter + 1) * step])
            counter += 1
        new_real = np.asarray(new_real)
        return new_real, new_index

    "y normalization"
    def normalize_y(self, data):
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
        return n_array

class alignment:
    def __init__(self, real_file, ncsimul_file):
        self.real_file = real_file
        self.ncsimul_file = ncsimul_file

        df_ncsimul= pandas.read_csv(filepath_or_buffer = adr+"NCSIMUL/"+self.ncsimul_file, sep = ';')
        self.ncsimul_ = df_ncsimul.values
        "Pc2"
        self.ncsimul = self.ncsimul_[:,56]
        self.ncsimul_index_ = range(len(self.ncsimul))

        df_real = pandas.read_csv(filepath_or_buffer = adr + "Real/"+ self.real_file, sep= ';')
        real_ = df_real.values
        'first column'
        self.real = real_[:,0]
        self.real_index_ = range(len(self.real))

        self.normali = normalise()

    def remove_const(self, input, limit):
        filter_ = []
        index_ = []
        i = 0
        for j in input:
            if j > limit:
                filter_.append(j)
                index_.append(i)
            i += 1
        return filter_, index_

    def find_groups(self, input_indexes):
        groups = []
        changed_index = [0]
        for i in range(len(input_indexes) - 1):
            if input_indexes[i] + 1 != input_indexes[i + 1]:
                # print(input_indexes[i], input_indexes[i+1])
                groups.append([input_indexes[i], input_indexes[i + 1]])
                changed_index.append(i)
                changed_index.append(i + 1)
        changed_index.append(len(input_indexes) - 1)
        # print(changed_index)
        return groups

    def prepare_remove(self, real_threshold, nc_threshold):

        filter_real, real_index = self.remove_const(self.real, real_threshold)
        filter_ncsimul, ncsimul_index = self.remove_const(self.ncsimul, nc_threshold)
        groups_ncsimul = self.find_groups(ncsimul_index)
        groups_real = self.find_groups(real_index)

        # print(groups_ncsimul)
        # print(groups_real)
        print(len(groups_ncsimul))
        print(len(groups_real))

        plt.plot(self.real)
        plt.plot([real_threshold] * len(self.real), "--", color="red")
        plt.show()

        plt.plot(self.ncsimul)
        plt.plot([nc_threshold] * len(self.ncsimul), "--", color="red")
        plt.show()

        return filter_real, real_index, filter_ncsimul, ncsimul_index

    def prepare_smooth_x(self, input_real, input_ncsimul):
        window = round(len(input_real) / len(input_ncsimul))
        input_smooth, indexes = self.normali.average_x(input_real, window)
        return input_smooth, indexes

    def prepare_normal_y(self, real_, ncsimul_):
        normal_ncsimul = self.normali.normalize_y(ncsimul_)
        normal_real = self.normali.normalize_y(real_)

        return normal_real, normal_ncsimul

    def prep_dtw(self, normal_real_, normal_ncsimul_, min, max):

        path = dtw.warping_path(normal_real_[min:max], normal_ncsimul_[min:max])
        distance, paths = dtw.warping_paths(normal_real_[min:max], normal_ncsimul_[min:max])
        dtwvis.plot_warping(normal_real_[min:max], normal_ncsimul_[min:max], path, filename="warp_results.png")

        best_path = dtw.best_path(paths)
        dtwvis.plot_warpingpaths(normal_real_[min:max], normal_ncsimul_[min:max], paths, best_path, filename="best_path_results.png")

        return path, distance

    def concatenate_lists(self, index_list):
        mylist = [index_list[0][0], index_list[-1][1]]
        return mylist

    def real_mapping_ncsimul_indexes(self, path_, real_indexes):
        real_to_ncsimul_index = {}
        for x, y in path_:
            if y in real_to_ncsimul_index:
                real_to_ncsimul_index[y].append(real_indexes[x])
            else:
                real_to_ncsimul_index[y] = [real_indexes[x]]

        _real_to_ncsimul = {i: self.concatenate_lists(j) for (i, j) in real_to_ncsimul_index.items()}

        return real_to_ncsimul_index

alg = alignment(real_file, ncsimul_file)
real_, real_index_, ncsimul, ncsimul_index = alg.prepare_remove(300,10)
real, real_index_after_smooth = alg.prepare_smooth_x(real_, ncsimul)
normal_real, normal_ncsimul = alg.prepare_normal_y(real, ncsimul)

plt.plot(normal_real, color='blue')
plt.plot(normal_ncsimul, color='red')
plt.show()

pathi, distanci = alg.prep_dtw(normal_real, normal_ncsimul, 0, 100)
print(alg.real_mapping_ncsimul_indexes(pathi, real_index_after_smooth))

"""other codes. TWe don't use them at the moment."""
def filter_frequency(input):

    fourier = np.fft.fft(input)
    n = input.size
    freq = np.fft.fftfreq(n)
    return freq

def filter_diff(input):
    _d = np.diff(input)
    return _d

def filter(input, epsilon, which_):
    x_ = [i for i in range(len(input))]
    try:
        which_ in ['diff', 'freq']
    except TypeError:
        print('the selected filter type not exist')
    else:
        if which_ == 'diff':
            _d = filter_diff(input)
        elif which_ == 'freq':
            _d = filter_frequency(input)

    #print(_d)
    n = len(_d)
    plt.plot(x_[0:n], input[0:n])
    plt.show()

    x_filterd = [i for i in range(len(_d)) if np.abs(_d[i])<epsilon ]
    #print(len(x_filterd))
    #print(x_filterd)

    plt.scatter(x_real[0:n], _d[0:n])
    plt.plot(x_[0:n], [0]*n)
    plt.plot(x_[0:n], [epsilon]*n)
    plt.plot(x_[0:n], [-1*epsilon]*n)
    plt.show()

    plt.scatter(x_filterd, [_d[j] for j in x_filterd])
    plt.show()

    print(len(x_filterd)/len(input))

    n2 = 1000 #len(input)
    plt.plot(x_[0:n2], input[0:n2])
    for j in x_filterd:
        if j < n2:
            plt.axvline(x=j, color='gray')
        else:
            break
    plt.show()
    return

#filter(real, 0.0001, 'diff')

# x = real
# peaks, _ = find_peaks(x, height=300)
# plt.plot(x)
# plt.plot(peaks, x[peaks], "x")
# plt.plot(np.zeros_like(x), "--", color="gray")
# plt.show()
#
# plt.plot(x[peaks])
# plt.show()






