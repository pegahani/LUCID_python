import pickle
import pandas
import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model


adress = "../../Data/energy_data/CMD/"

"""["Nom de l'essai", 'Vc, m/min', 'Fz, mm/dent/tr', 'Lu, mm', 'ae, mm', 'ap, mm', 'P0_avg, W', 'Pt_avg, W', 'Pc_avg, W', 'Wc, W/cm3/min', 'Dc, mm', 'Rb, mm', 'Nb dent']"""

#df_power  = pandas.read_excel(input, parse_cols = range(9))
#df_params = pandas.read_excel(input, parse_cols = range(11,24))

class file_:

    def __init__(self, name_, adress):

        self.name = adress + name_
        self.file_adress= adress + name_ + ".xlsx"
        self.adress = adress

    def upload_params(self, range_, index_):

        df = pandas.read_excel(self.file_adress, usecols=range_)
        array_output = []

        for index, row in df.iterrows():
            if index > index_:
                break

            array_output.append([item for item in row][1:])

        return np.asarray(array_output)

    def upload_powers(self, range_):
        list_output = []
        df_power = pandas.read_excel(self.file_adress, usecols= range_)
        for i in range(0,len(range_)):
            list_output.append(df_power.iloc[:, i])

        return np.asarray(list_output)

    def save_samples(self, params_range, parmas_index, powers_range):

        params = self.upload_params(params_range, parmas_index,)
        powers = self.upload_powers(powers_range)

        with open(self.name + '.pickle', 'wb') as handle:
            pickle.dump(params, handle)
            pickle.dump(powers, handle)

        return

    def load_samples(self):
        with open(self.name + '.pickle', 'rb') as handle:
            params = pickle.load(handle)
            powers = pickle.load(handle)

        return (params, powers)

    def save_separate(self):
        with open(self.name + '.pickle', 'rb') as handle:
            params = pickle.load(handle)
            powers = pickle.load(handle)

        with open(self.name + '_params.pickle', 'wb') as handle:
            pickle.dump(params, handle)

        with open(self.name + '_powers.pickle', 'wb') as handle:
            pickle.dump(powers, handle)

        return

# example = file_("40CMD8_COM_1-01-01")
# tt = 6
# example.save_samples(params_range= range(tt+2, tt+2+13), parmas_index=4, powers_range=range(tt))
#
# toto = example.load_samples()
# print(toto[0].shape)
# print(toto[1].shape)
#
#
# example.save_separate()




