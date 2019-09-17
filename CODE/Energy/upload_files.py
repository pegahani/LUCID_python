# -*- coding: utf-8 -*-

import pickle
import pandas
import numpy as np
import csv

import matplotlib.pyplot as plt
from sklearn import linear_model


adress = "../../Data/energy_data/sac_data/5axes/NCSIMUL/"

"""["Nom de l'essai", 'Vc, m/min', 'Fz, mm/dent/tr', 'Lu, mm', 'ae, mm', 'ap, mm', 'P0_avg, W', 'Pt_avg, W', 'Pc_avg, W', 'Wc, W/cm3/min', 'Dc, mm', 'Rb, mm', 'Nb dent']"""

#df_power  = pandas.read_excel(input, parse_cols = range(9))
#df_params = pandas.read_excel(input, parse_cols = range(11,24))

class file_:

    def __init__(self, name_, adress, file_type):

        self.name = adress + name_
        if file_type == "xlsx":
            self.file_adress= adress + name_ + ".xlsx"
        elif file_type == 'csv':
            self.file_adress= adress + name_ + ".csv"
        self.adress = adress

    def save_csv(self):
        df = pandas.read_csv(self.file_adress,sep=';',encoding= 'latin-1')
        #df = df.values
        with open(self.name + '.pickle', 'wb') as handle:
            pickle.dump(df, handle)
        return

    def load_pickles(self):
        with open(self.name + '.pickle', 'rb') as handle:
            df = pickle.load(handle)
        return df

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

# example = file_("Projet-Lucid-ACIER_stats",adress, 'csv')
df = pandas.read_csv(adress+"LUCID_5axes-Acier_Hermle_stats.csv",sep=';',encoding= 'latin-1')
#
# "indexes of changed strings in ToolRef column (35th column)"
tempo = df['ToolRef'].ne(df['ToolRef'].shift().bfill())
result = np.where(tempo == True)
changed_indexes = result[0]
print(changed_indexes)
#
# #writer = csv.writer(open(adress+"too1_Acier_S_real.csv", 'w'))
# #data_to_csv = df.head(n= changed_indexes[0])
# data_to_csv = df[changed_indexes[2]:]
# data_to_csv.to_csv(adress+"ACIER_S_4_ncsimul.csv", sep=';', encoding='utf-8', header=0)

# df_ncsimul= pandas.read_csv(adress+"ACIER_S_4_ncsimul_FraiseMtsubishiVC2PSBR0400.csv", sep=';',encoding= 'latin-1')
# ncsimul = df_ncsimul.values
# #random.shuffle(ncsimul)
# ncsimul_ = ncsimul[:, 56]
# print(ncsimul_)
# with open(adress + 'ACIER_S_ncsimul_4.pickle', 'wb') as handle:
#     pickle.dump(ncsimul_, handle)


