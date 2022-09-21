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








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inn1', help='model')
    parser.add_argument('inn2', help='data')
    parser.add_argument('out', help='output')


    opts = parser.parse_args()

    model = load_model(opts,inn1)
    input = np.loadtxt(opts.inn2, skiprows=1, dtype=np.int8)

    prediction = model.predict(input)
    print(prediction)