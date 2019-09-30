import csv
import itertools
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

adr = "../../Data/energy_data/sac_data/GP2R/"
# ncsimul_file = "ACIER_S_1_ncsimul_F4042R.T22.025.Z03.10.csv"
# real_file = "ACIER_S_1_real_F4042R.T22.025.Z03.10.csv"

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

        comparison = int(len(input_curve)/step)*step
        while counter * step < comparison:
            new_real.append(np.mean(input_curve[counter * step:(counter + 1) * step]))
            new_index.append([counter * step, (counter + 1) * step])
            counter += 1
        new_index.append([counter * step, len(input_curve)])
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
    def __init__(self, real_, ncsimul_, only_dtw):

        if only_dtw:
            self.ncsimul = ncsimul_
            self.real = real_
        else:
            "PC2"
            self.ncsimul_ = ncsimul_
            self.ncsimul = ncsimul_[:,56]

            'first column'
            self.real_ = real_
            self.real = real_[:, 0]

        self.ncsimul_index_ = range(len(self.ncsimul))
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
        return groups

    def prepare_remove(self, real_threshold, nc_threshold):

        filter_real, real_index = self.remove_const(self.real, real_threshold)
        filter_ncsimul, ncsimul_index = self.remove_const(self.ncsimul, nc_threshold)
        groups_ncsimul = self.find_groups(ncsimul_index)
        groups_real = self.find_groups(real_index)

        # print(groups_ncsimul)
        # print(groups_real)

        plt.plot(self.real)
        plt.plot([real_threshold] * len(self.real), "--", color="red")
        plt.show()

        plt.plot(self.ncsimul)
        plt.plot([nc_threshold] * len(self.ncsimul), "--", color="red")
        plt.show()

        return filter_real, real_index, filter_ncsimul, ncsimul_index

    def prepare_smooth_x(self, input_real, input_ncsimul, given_window = False):

        input_smooth_real, indexes_real, input_smooth_ncsimul, indexes_ncsimul = None, None, None, None
        if not given_window:
            window = round(len(input_real) / len(input_ncsimul))
        else:
            window = round(len(input_real)/3000)
            input_smooth_ncsimul, indexes_ncsimul = self.normali.average_x(input_ncsimul, window)

        input_smooth_real, indexes_real = self.normali.average_x(input_real, window)

        return input_smooth_real, indexes_real, input_smooth_ncsimul, indexes_ncsimul

    def prepare_normal_y(self, real_, ncsimul_):
        normal_ncsimul = self.normali.normalize_y(ncsimul_)
        normal_real = self.normali.normalize_y(real_)

        return normal_real, normal_ncsimul

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

    def replot_after_dtw(self, ncsimul_to_real_, real_index__, ncsimul_index_, only_dtw):
        new_nc = []
        new_real = []

        if not only_dtw:
            for (key, values) in ncsimul_to_real_.items():
                new_nc.append(ncsimul_index_[key])
                new_real.append(np.mean(list(itertools.chain(
                    [[self.real[real_index__[item[0]]], self.real[real_index__[item[-1]]]]
                        for item in values]))))
                # new_y.append(np.mean([origin_real[real_index_[values[0][0]]], origin_real[real_index_[values[-1][-1]]]]))

        elif only_dtw:
            for (key, values) in ncsimul_to_real_.items():
                equiv_interval = ncsimul_index_[key]
                new_nc.append(list(itertools.chain(equiv_interval)))
                new_real.append(np.mean(list(itertools.chain(
                    [[self.real[real_index__[item[0]]], self.real[real_index__[item[-1]]]]
                        for item in values]))))

        plt.plot(new_nc, new_real)
        plt.plot(self.ncsimul)
        plt.show()

        plt.plot(new_nc, self.normali.normalize_y(new_real))
        plt.plot(self.normali.normalize_y((self.ncsimul)))
        plt.show()
        return

    def re_assign_after_alignment(self, ncsimul_to_real_, real_index__, ncsimul_index_, ncsimul_index_after_smooth, only_dtw = False):
        new_nc = []
        new_real = []
        nc_equiv = None

        for (key, values) in ncsimul_to_real_.items():
            if not only_dtw:
                nc_equiv = ncsimul_index_[key]
            else:
                equiv_interval = ncsimul_index_after_smooth[key]
                tempo2 = []
                tempo2 = tempo2 + [ncsimul_index_[j] for j in range(equiv_interval[0],equiv_interval[-1])]# equiv_interval[-1])]
                new_nc = new_nc + tempo2

            tempo = []
            for item in values:
                tempo = tempo + [real_index__[i] for i in range(item[0], item[-1])]#item[-1])]
            new_real = new_real + tempo

            if not only_dtw:
                new_nc = new_nc + [nc_equiv]*len(tempo)

        return new_nc, new_real

    def joint_real_to_ncsimul(self, material, sequence, new_nc, new_real):

        # plt.plot([self.real[j] for j in new_real], color='blue')
        # plt.plot([self.ncsimul[k] for k in new_nc], color = 'red')
        # plt.show()

        with open(adr + 'Juncture/' + str(material) + '_' + str(sequence)+ '.csv', mode='w') as _file:
            _writer = csv.writer(_file, delimiter=';')#, quotechar='"', quoting=csv.QUOTE_MINIMAL)

            _writer.writerow(['N Brut',	'N Outil',	'N Bloc',	'Cycle usinage','Tabs',	'T', 'Trel',
                                  'Vc', 'fz', 'h', 'Ae', 'Ap', 'AD', 'Du', 'V',	'Angle', 'Q', 'Temps Outils', 'Sequence outil', 'AxisCoordX',
                                 'AxisCoordY', 'AxisCoordZ', 'A', 'B', 'C', 'TcpX',	'TcpY',	'TcpZ',	'ToolAxisX', 'ToolAxisY', 'ToolAxisZ', 'S',	'TypeIntersect',
                                 'TypeMovement', 'TypeInOut', 'ToolRef', 'ToolFamilly',	'ToolType',	'Dc', 'Lc',	'Rb', 'F',	'Record1',	'Record2', 'RecordPassant',
                                 'RecordPrecPassant', 'DataPhase',	'InteractMode',	'ContactMode',	'AeEquiv',	'ApEquiv',	'Wc1',	'Pc1',	'Tc1',
                                 'Ft1',	'Wc2',	'Pc2',	'Tc2',	'Ft2', 'Pcreal'])

            try:
                len(new_nc) == len(new_real)
            except:
                raise NameError('the new_real and new_nc should have the same length')
            else:
                print(len(new_nc))
                for i in range(len(new_real)):
                    l = list(self.ncsimul_[new_nc[i]])
                    l.append(self.real_[new_real[i],0])
                    _writer.writerow(l)
        return




