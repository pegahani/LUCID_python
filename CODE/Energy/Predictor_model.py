import pickle
import random
import numpy
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from numpy import loadtxt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import transformations as trafo
from keras import layers, optimizers

from time_wrapping import normal_dtw

#real data prepared by Phillipe
adr_COM = "../../Data/energy_data/phillipe/UF1_ COM_ACIER_T1_Train.xlsx"

class predictions:
    def __init__(self):
        df = pandas.read_excel(adr_COM)
        df_ar_tp = df.values
        self.df_COM = df_ar_tp[0:47, ]

    def prep_COM(self):
        "parameters for training: Ap, Ae, F, Z and test: Pc2"
        self.x_tarin_COM = self.df_COM[:, 0:4]
        y_train = self.df_COM[:, 6]
        self.y_train_COM = y_train.reshape(47, 1)
        return

    def train_LR(self,x_tarin,  y_train):
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(x_tarin, y_train)
        return regr

    def predict_on_LR(self, x_test, y_test, regr_):
        _y_pred = regr_.predict(x_test)
        plt.plot(_y_pred)
        plt.plot(y_test)
        plt.show()
        return _y_pred


# df_ncsimul= pandas.read_csv("../../Data/energy_data/phillipe/Projet-Lucid-ACIER_stats_Eval.csv")
# ncsimul = df_ncsimul.values
# #random.shuffle(ncsimul)
# ncsimul_x_test = ncsimul[:, 0:4]
# ncsimul_y_test = ncsimul[:, 7]
# ncsimul_y_real = ncsimul_y_test.copy()
#
# # with open('ncsimul_data.pickle', 'wb') as handle:
# #     pickle.dump(ncsimul_y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('ncsimul_data.pickle', 'rb') as handle:
#     ncsimul_y_test = pickle.load(handle)
#
# y_ = [i for i in range(len(ncsimul_y_test))]
#
# print('*****************')
#
#
# y__ = [i for i in range(len(y_train))]
# #plt.plot(y__, y_train)
# # plt.show()
#
# # Create linear regression object
# regr = linear_model.LinearRegression()
# # Train the model using the training sets
# regr.fit(ncsimul_x_test, ncsimul_y_test)
# _y_pred = regr.predict(ncsimul_x_test)
# # plt.title("Phillippe data")
# # plt.plot(y_, _y_pred)
# # plt.plot(y_, ncsimul_y_test)
# # plt.show()
#
# adr = "../../Data/energy_data/new/synthese_essai_COM_v2.xlsx"
# df = pandas.read_excel(adr)
# df_ar = df.values
#
# print('*******************')
# ncsimul_y_ = ncsimul_y_test.reshape(30639,1)
# x_tarin = x_tarin.astype(np.float64)
#
# normed_ncsimul_x_test = trafo.unit_vector(ncsimul_x_test, axis=1)
# normed_x_train = trafo.unit_vector(x_tarin, axis=1)
#
# y__ = [i for i in range(len(ncsimul_y_))]
# #plt.plot(y__, ncsimul_y_)
# # plt.show()
#
# # fix random seed for reproducibility
# numpy.random.seed(7)
#
# # # define the keras model
# model = Sequential()
# model.add(Dense(64, input_shape=(4,), activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))
# #%%%%%%%%%%
# #model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse'])
# #model.fit(normed_ncsimul_x_test, ncsimul_y_, epochs=2000, batch_size=128)
# #model.save_weights("model_new.h5")
# model.load_weights("model_new.h5")
#
# test_predictions = model.predict(normed_ncsimul_x_test)#.flatten()
# ##############
# exp = normal_dtw(test_predictions, 0)
# exp.average_x()
# exp.total_normalisation()
#
# plt.plot(exp.new_real_x, exp.new_real)
# plt.plot(exp.ncsimul_x_test, exp.ncsimul)
# plt.show()
#
# plt.plot(exp.new_real_x, exp.new_real_normal)
# plt.plot(exp.ncsimul_x_test, exp.ncsimul_y_normal)
# plt.show()
#
# exp.dtw_(length_min=0, length_max=5000)
# ##############
#
# all_layers = model.layers
# for layer in model.layers[0:3]:
#     layer.trainable = False
#
# # Create the new model
# model_plus = Sequential()
# model_plus.add(model)
#
# # Add new layers
# #model_plus.add(layers.Flatten())
#
# model_plus.add(Dense(32, activation='relu'))
# model_plus.add(Dense(32, activation='relu'))
# model_plus.add(layers.Dense(1))
# model_plus.summary()
#
# # model_plus.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse'])
# # model_plus.fit(normed_x_train, y_train, epochs=2000, batch_size=256)
# # Save the model
# # model_plus.save('model_plus.h5')
# model_plus.load_weights('model_plus.h5')
#
# predictions_after_real = model_plus.predict(normed_ncsimul_x_test)#.flatten()
# exp_after_real = normal_dtw(predictions_after_real, 100)
# exp_after_real.average_x()
# exp_after_real.total_normalisation()
#
# plt.plot(exp_after_real.new_real_x, exp_after_real.new_real)
# plt.plot(exp_after_real.ncsimul_x_test, exp_after_real.ncsimul)
# plt.show()
#
# plt.plot(exp_after_real.new_real_x, exp_after_real.new_real_normal)
# plt.plot(exp_after_real.ncsimul_x_test, exp_after_real.ncsimul_y_normal)
# plt.show()
#
# exp_after_real.dtw_(length_min=0, length_max=50)
# # plt.plot(y_, ncsimul_y_real)
# # plt.plot(y_, test_predictions)
# # plt.show()
#
#
# # print('********')
# # TA_1S = df_ar[df_ar[:,0]== 'TA1S']
# # print('TA_1S')
# # print(TA_1S.shape)
# #
# # x_tarin = TA_1S[:,4:8]
# # y_train = TA_1S[:, 16]
# # # Create linear regression object
# # regr = linear_model.LinearRegression()
# # # Train the model using the training sets
# # regr.fit(x_tarin, y_train)
# # _y_pred = regr.predict(ncsimul_x_test)
# # plt.title("TA_1S")
# # plt.plot(y_, _y_pred)
# #
# # plt.plot(y_, ncsimul_y_test)
# # plt.show()



