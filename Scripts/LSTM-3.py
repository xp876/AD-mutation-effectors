import tensorflow as tf
print(tf.__version__)

#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Input, Dense, Flatten, Add, Lambda, LeakyReLU, Embedding,Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib
matplotlib.use('Agg')
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
import re
import sys
import os,glob
import numpy as np
import os
import sys
import argparse
import random
import re
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr
from scipy import optimize as op
from sklearn.inspection import permutation_importance
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Add, Lambda, LeakyReLU, Embedding
from tensorflow.keras.layers import Dropout, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, SpatialDropout1D
from tensorflow.keras.layers import GRU, Activation, Bidirectional, LSTM, concatenate, BatchNormalization,TimeDistributed
from tensorflow.keras.optimizers import SGD, Adam
from functools import partial
from bayes_opt import BayesianOptimization
import keras_metrics as km
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from keras.models import Sequential, Model
# from keras.layers.convolutional import MaxPooling2D, Convolution2D
# from keras.layers.core import Dense, Dropout, Activation, Flatten
import json
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import binary_accuracy,Precision,Recall,AUC
from tensorflow.keras.metrics import FalseNegatives,FalsePositives,TrueNegatives,TruePositives
from tensorflow.keras.metrics import MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,RootMeanSquaredError
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare
from sklearn.model_selection import StratifiedKFold, KFold,RepeatedKFold
from keras.optimizers import SGD, Adam
from functools import partial
from bayes_opt import BayesianOptimization
from tensorflow.keras import metrics

from tensorflow.keras import regularizers










def create_model(dropout_rate_1=0.0,uts_1=1,uts_2=1,acti_1=1,
                 acti_2=1,
                 init_mode_1=1,
                 init_mode_2=1,
                 init_mode_3=1,
                 ll_1=1,ll_2=1,ll_3=1):
        # create model
    input_shape=(X_train.shape[1],)
    inputs=Input(shape=input_shape)

    dropout_rate_1 = round(dropout_rate_1, 1)



    if 1 <= uts_1 <= 2:
        uts_1 = 256
    elif 2 < uts_1 <= 3:
        uts_1 = 128
    elif 3 < uts_1 <= 4:
        uts_1 = 64
    elif 4 < uts_1 <= 5:
        uts_1 = 32
    elif 5 < uts_1 <= 6:
        uts_1 = 16        

    if 1 <= uts_2 <= 2:
        uts_2 = 256
    elif 2 < uts_2 <= 3:
        uts_2 = 128
    elif 3 < uts_2 <= 4:
        uts_2 = 64
    elif 4 < uts_2 <= 5:
        uts_2 = 32
    elif 5 < uts_2 <= 6:
        uts_2 = 16    
        





    if uts_2>uts_1:
        uts_2=uts_1
        





    if 1 <= acti_1 <= 2:
        acti_1 = 'elu'
    elif 2 < acti_1 <= 3:
        acti_1 = 'relu'
    elif 3 < acti_1 <= 4:
        acti_1 = 'tanh'

    if 1 <= acti_2 <= 2:
        acti_2 = 'elu'
    elif 2 < acti_2 <= 3:
        acti_2 = 'relu'
    elif 3 < acti_2 <= 4:
        acti_2 = 'tanh'



    if 1 <= init_mode_1 <= 2:
        init_mode_1 = 'normal'
    elif 2 < init_mode_1 <= 3:
        init_mode_1 = 'ones'
    elif 3 < init_mode_1 <= 4:
        init_mode_1 = 'uniform'

    if 1 <= init_mode_2 <= 2:
        init_mode_2 = 'normal'
    elif 2 < init_mode_2 <= 3:
        init_mode_2 = 'ones'
    elif 3 < init_mode_2 <= 4:
        init_mode_2 = 'uniform'

    if 1 <= init_mode_3 <= 2:
        init_mode_3 = 'normal'
    elif 2 < init_mode_3 <= 3:
        init_mode_3 = 'ones'
    elif 3 < init_mode_3 <= 4:
        init_mode_3 = 'uniform'



    if 1 <= ll_1 <= 2:
        ll_1 = 1e-4
    elif 2 < ll_1 <= 3:
        ll_1 = 1e-3
    elif 3 < ll_1 <= 4:
        ll_1 = 1e-2

    if 1 <= ll_2 <= 2:
        ll_2 = 1e-4
    elif 2 < ll_2 <= 3:
        ll_2 = 1e-3
    elif 3 < ll_2 <= 4:
        ll_2 = 1e-2
        
    if 1 <= ll_3 <= 2:
        ll_3 = 1e-4
    elif 2 < ll_3 <= 3:
        ll_3 = 1e-3
    elif 3 < ll_3 <= 4:
        ll_3 = 1e-2
        


    ds_1 = GRU(units=uts_1, kernel_initializer=init_mode_1, activation=acti_1,kernel_regularizer=regularizers.l1_l2(l1=ll_1, l2=ll_1))(inputs)
    bn_1=BatchNormalization()(ds_1)
    dp_1=Dropout(dropout_rate_1)(bn_1)
    ds_2 = Dense(units=uts_2, kernel_initializer=init_mode_2, activation=acti_2,kernel_regularizer=regularizers.l1_l2(l1=ll_2, l2=ll_2))(dp_1)


    ds_3= Dense(units=1, kernel_initializer=init_mode_3,kernel_regularizer=regularizers.l1_l2(l1=ll_3, l2=ll_3))(ds_2)

    model = Model(inputs=inputs, outputs=ds_3)


    print("dropout_rate_1----"  +
          "uts_1----" +"uts_2----" +
          "acti_1----" +"acti_2----" +
           "init_mode_1----" + "init_mode_2----" + "init_mode_3----" +
          "ll_1----"+"ll_2----"+"ll_3----")

    print(str(dropout_rate_1) + "----"  +
          str(uts_1) + "----"  +str(uts_2) + "----"  +
          str(acti_1) + "----"  +str(acti_2) + "----"  +
          str(init_mode_1) + "----"  +str(init_mode_2) + "----"  +str(init_mode_3) + "----"  +
          str(ll_1) + "----"  +str(ll_2) + "----"  +str(ll_3) + "----"  )
          


    #model.compile(loss='mean_squared_error', optimizer='adam')
    return model









def fit_with(epochs, bs, dropout_rate_1,uts_1,uts_2,acti_1,
             acti_2,
             init_mode_1,init_mode_2,init_mode_3,
             ll_1,ll_2,ll_3):
    # Create the model using a specified hyperparameters.

    for i, (train, test) in enumerate(rkf.split(X_train, Y_train)):
        print('\n\n%d' % i)

        model = create_model(dropout_rate_1,uts_1,uts_2,acti_1,
             acti_2,
             init_mode_1,init_mode_2,init_mode_3,
             ll_1,ll_2,ll_3)
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=[metrics.MeanSquaredError(), metrics.RootMeanSquaredError(),
                               metrics.MeanAbsoluteError()])
    # model.summary()
        print('Train...')
        
        #round(dropout_rate_1, 4)

        epochs = int(epochs)




            # bs=int(bs)
        if 1 <= bs <= 2:
            bs = 8
        elif 2 < bs <= 3:
            bs = 16
        elif 3 < bs <= 4:
            bs = 32
        elif 4 < bs <= 5:
            bs = 64

        mc = ModelCheckpoint(str(round(epochs, 4)) + "-"  +str(round(bs, 4)) + "-"  +
          str(round(dropout_rate_1, 4)) + "-"  +
          str(round(uts_1, 4)) + "-"  +str(round(uts_2, 4)) + "-"  +
          str(round(acti_1, 4)) + "-"  +str(round(acti_2, 4)) + "-"  +
          str(round(init_mode_1, 4)) + "-"  +str(round(init_mode_2, 4)) + "-"  +str(round(init_mode_3, 4)) + "-"  +
          str(round(ll_1, 4)) + "-"  +str(round(ll_2, 4)) + "-"  +str(round(ll_3, 4))  + "-"+'.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    
    
#        mc = ModelCheckpoint(str(epochs) + "-"  +str(bs) + "-"  +
#          str(dropout_rate_1) + "-"  +str(dropout_rate_2) + "-"  +str(dropout_rate_3) + "-"  +
#          str(uts_1) + "-"  +str(uts_2) + "-"  +str(uts_3) + "-"  +str(uts_4) + "-"  +
#          str(acti_1) + "-"  +str(acti_2) + "-"  +str(acti_3) + "-"  +str(acti_4) + "-"  +
#          str(init_mode_1) + "-"  +str(init_mode_2) + "-"  +str(init_mode_3) + "-"  +str(init_mode_4) + "-"  +str(init_mode_5) + "-"  +
#          str(ll_1) + "-"  +str(ll_2) + "-"  +str(ll_3) + "-"  +str(ll_4) + "-"  +str(ll_5) + "-"+'.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_mean_absolute_error', mode='auto', verbose=1, patience=10,restore_best_weights=True)



    # Train the model with the train dataset.
        history = model.fit(x=X_train[train], y=Y_train[train], epochs=epochs,
                        batch_size=bs, validation_data = (X_train[test], Y_train[test]), shuffle=True, verbose=1,
                        callbacks=[mc,es])

        score = model.evaluate(X_train[test], Y_train[test], batch_size=bs, verbose=1)

        mse_score.append(score[1])
        mae_score.append(score[3])
        rmse_score.append(score[2])

    # score = model.evaluate(x=X_test, y=Y_test, batch_size=bs, verbose=2)

        print(model.metrics_names)
        print(score)


# print(score[2])
# print('Test loss:', score)


    print('***********************print final result*****************************')
    mean_mse = np.mean(mse_score)
    mean_mae = np.mean(mae_score)
    mean_rmse = np.mean(rmse_score)
    print('epoch'+'\t'+'batch size'+'\t'+str(epochs)+'\t'+str(bs))
    print("mean_mse"+'\t'+"mean_mae"+'\t'+"mean_rmse"+'\t'+str(mean_mse)+'\t'+str(mean_mae)+'\t'+str(mean_rmse))


    return -mean_mae




import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='data')
    parser.add_argument('targetfile', help='target')


    parser.add_argument('ww', help='width')
    parser.add_argument('dd', help='depth')
    parser.add_argument('rr', help='random')
    opts = parser.parse_args()


    width=int(opts.ww)
    depth=int(opts.dd)
    ran=int(opts.rr)

    mat = np.loadtxt(opts.datafile)
    label = np.loadtxt(opts.targetfile, dtype=np.int8)
    print(label.shape)
    print(mat.shape)

    seed = 7
    validation_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(mat, label, test_size=validation_size, random_state=seed)

    # np.savetxt('/stor/work/SongYi/xp876/ad-data/gen/10000/prediction/X_train.txt', X_train, delimiter='\t',fmt='%i')
    # np.savetxt('/stor/work/SongYi/xp876/ad-data/gen/10000/prediction/X_test.txt', X_test, delimiter='\t',fmt='%i')
    # np.savetxt('/stor/work/SongYi/xp876/ad-data/gen/10000/prediction/Y_train.txt', Y_train, delimiter='\t')
    # np.savetxt('/stor/work/SongYi/xp876/ad-data/gen/10000/prediction/Y_test.txt', Y_test, delimiter='\t')

    print(X_test.shape)
    print(Y_test.shape)
    print(X_train.shape)
    print(Y_train.shape)

    #X_train = mat[:50, :1000]
    #Y_train = label[:50]

    mse_score = []
    mae_score = []
    rmse_score = []
    # kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=ran)


    training_start_time = time.time()
    print(training_start_time)

    fit_with_partial = partial(fit_with)

    pbounds = {'epochs': (50, 50.99), 'bs': (1, 4.99), 'dropout_rate_1': (0, 0.549),
           'uts_1': (1, 5.99), 'uts_2': (1, 5.99),
            'acti_1': (1, 3.99),'acti_2': (1, 3.99),
           'init_mode_1': (1, 3.99), 'init_mode_2': (1, 3.99),'init_mode_3': (1, 3.99),
          'll_1': (1, 3.99),'ll_2': (1, 3.99),'ll_3': (1, 3.99),
          }

    optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
    optimizer.maximize(
    init_points=width,
    n_iter=depth,
)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))




