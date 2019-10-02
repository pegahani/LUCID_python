from __future__ import absolute_import, division, print_function, unicode_literals

from keras import Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random
import pathlib
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import numpy as np
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout, Concatenate
from keras.models import Sequential
from keras.layers import Dense

column_names = ['Ae','Ap','F','S','V','AD','Angle','fz','Vc','Q','h','InteractMode','ContactMode','AeEquiv','ApEquiv','TcpX','TcpY','TcpZ','Pc2', 'Pcreal']
source_com = ['F', 'S', 'AE', 'AP']

class learn_nn:
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
        pass

    def split_train_test(self, raw_dataset, percentage):

        #normalized_data=(raw_dataset-raw_dataset.mean())/raw_dataset.std()

        ae_ap = raw_dataset.Ae * raw_dataset.Ap * 5
        raw_dataset['ae_ap'] = ae_ap

        #ae_ap_fz = raw_dataset.Ae * raw_dataset.Ap * raw_dataset.fz
        #raw_dataset['ae_ap_fz'] = ae_ap_fz

        #dataf=((self.raw_dataset-self.raw_dataset.min())/(self.raw_dataset.max()-self.raw_dataset.min()))
        #dataf['Pc2']=self.raw_dataset['Pc2']
        #dataf['S']=self.raw_dataset['S']

        # print('************')
        # for i in self.raw_dataset['RecordPassant']:
        #     if str(i).__contains__('°'):
        #         print(i)

        train, test = train_test_split(raw_dataset, test_size= percentage, shuffle=True)
        return train, test

    def train_on_nn(self, train, test):

        source_data = column_names[:15] #['Ae','Ap','F','S','AD','Angle','fz','Vc','V','Q','h','InteractMode','ContactMode','AeEquiv','ApEquiv']
        #source_data = ['Ae','Ap','F','S','AD','Angle','fz','Vc','V','Q','h','InteractMode','ContactMode','AeEquiv','ApEquiv','RecordPassant']
        #source_data = ['Ae','Ap']
        #source_data = ['ae_ap']
        #source_data = ['Ae','Ap','F','S','Q','Angle','fz','Vc','V','AD','h']
        #source_data = ['Ae','Ap','F','S']
        #source_data = ['Ae','Ap','F','S','apn']
        #ae * ap est utile
        #AD hyper important
        #Q variable la plus puissante
        #Q=AP*AE*Vf
        #Vf=fz*n*ZEFF
        #q=17.10 	5.22
        #ae=10.3	3.63
        #ap=3.5

        #target_data = ['Pc2']
        target_data = ['Pcreal']
        x_train = train[source_data]
        x_test = test[source_data]

        y_train = train[target_data]
        y_test = test[target_data]

        model = Sequential()
        model.add(Dense(12, input_dim=len(source_data),activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(6,activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(1,activation='relu'))
        #opt=Adam(lr=0.01)
        opt=Adam()
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        model.fit(x_train, y_train, epochs=20000, batch_size=128, validation_data=(x_test, y_test))

        #model.save("model.h5")
        y_predicted = model.predict(x_test, verbose=1)
        #print(predictedx-y_test, x_test)

        #create a dat
        res_data=pd.DataFrame()
        res_data[target_data] = y_test
        res_data['pred'] = y_predicted
        res_data['error'] = y_predicted - y_test
        res_data[source_data] = x_test
        res_data['rule'] = (res_data['Ae'] + 2 * res_data['Ap']) * 7
        #print(res_data)
        res_data.to_csv("res2.csv",sep=';')

        return

    def train_diff_nn(self, COM, juncture):

        source_data_1 = column_names[5:15] #['AD','Angle','fz','Vc','V','Q','h','InteractMode','ContactMode','AeEquiv','ApEquiv']
        source_data_2 = column_names[0:4] #['Ae', 'Ap', 'F', 'S']
        source_data = column_names[0:15] #['Ae', 'Ap', 'F', 'S', 'AD','Angle','fz','Vc','V','Q','h','InteractMode','ContactMode','AeEquiv','ApEquiv']
        com_target  = ['Pc']

        x_tarin_COM = COM[source_com]
        y_train_COM = COM[com_target]
        #y_train_COM = y_train.reshape(47, 1)

        train_junc, test_junc = train_test_split(juncture, test_size=0.2, shuffle=True)

        #x_train_juncture = np.column_stack((train_junc[:, 41], train_junc[:, 31], train_junc[:, 7:13],
        #                             train_junc[:, 14:17], train_junc[:, 47:51]))
        #y_train_junncture = train_junc[:, 56]

        date_target = ['Pc2']
        #target_data = ['Pcreal']
        x_train_1 = train_junc[source_data_1]
        y_train = train_junc[date_target]

        x_train_2 = train_junc[source_data_2]

        x_test = test_junc[source_data]
        y_test = test_junc[date_target]

        input_1 = Input(shape=(x_train_1.shape[1],))
        input_2 = Input(shape=(x_train_2.shape[1],))

        combined = Concatenate()([input_1, input_2])
        print(input_1.shape, input_2.shape, combined.shape)


        input_1 = keras.layers.Input(shape=(x_train_1.shape[1],))
        x1 = keras.layers.Dense(8, activation='relu')(input_1)
        #out_junc = keras.layers.Dense(8, activation='relu')(x2)
        #****
        input_2 = keras.layers.Input(shape=(x_train_2.shape[1],))
        x2 = keras.layers.Dense(8, activation='relu')(input_2)
        #out_com = keras.layers.Dense(8, activation='relu')(x1)

        # merging models
        added = keras.layers.Add()([x1, x2])

        # generate a model from the layers above
        out = keras.layers.Dense(1)(added)
        model = keras.models.Model(inputs=[input_1, input_2], outputs=out)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])

        # Always a good idea to verify it looks as you expect it to
        model.summary()

        # The resulting model can be fit with a single input:
        # model.fit(data, labels, epochs=50)

        print(x_train_1.shape)
        print(x_train_2.shape)
        print(y_train.shape)

        x_1 = np.array(x_train_1)
        x_2 = np.array(x_train_2)
        model.fit( x = [x_1, x_2], y = np.array(y_train), epochs=30, batch_size=128, validation_data=(x_test, y_test))

        pass

    def first_nn(self, datainput, source_data, target_data, percentage, weights_name):
        train, test = self.split_train_test(datainput, percentage)

        x_train = train[source_data]
        x_test = test[source_data]

        y_train = train[target_data]
        y_test = test[target_data]

        # print(y_test)

        model = Sequential()
        model.add(Dense(12, input_dim=len(source_data), activation='relu', name='dense_1'))
        # model.add(Dropout(0.2))
        model.add(Dense(6, activation='relu', name='dense_2'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu', name='dense_3'))
        # opt=Adam(lr=0.01)
        opt = Adam()
        # keras.losses.huber_loss(y_true, y_pred, delta=1.0 de 0.1 à 10)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])

        model.summary()

        model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))

        model.save_weights(weights_name+".h5")
        y_predicted = model.predict(x_test, verbose=1)
        # print(predictedx-y_test, x_test)

        # create a dat
        res_data = pd.DataFrame()
        res_data[target_data] = y_test
        res_data['pred'] = y_predicted
        res_data['error'] = y_predicted - y_test
        res_data[source_data] = x_test
        res_data['rule'] = (res_data['Ae'] + 2 * res_data['Ap']) * 7
        # print(res_data)
        res_data.to_csv("res2.csv", sep=';')

        pass

    def second_nn(self, data_input, source_data, target_data, percentage, given_weights):

        train, test = self.split_train_test(data_input, percentage)
        x_train = train[source_data]
        x_test = test[source_data]

        y_train = train[target_data]
        y_test = test[target_data]

        model = Sequential()
        model.add(Dense(12, input_dim=len(source_data), activation='relu', name='dense_1'))  # will be loaded
        model.add(Dense(12, activation='relu', name='new_dense_2'))  # will not be loaded
        model.add(Dense(6, activation='relu', name='new_dense_3'))  # will not be loaded
        model.add(Dense(1, activation='relu', name='new_dense_4'))

        model.load_weights(given_weights, by_name=True)

        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        # Always a good idea to verify it looks as you expect it to
        model.summary()

        # The resulting model can be fit with a single input:
        model.fit(x_train, y_train, epochs=300, batch_size=128, validation_data=(x_test, y_test))

        pass
