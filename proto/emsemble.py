#-*- coding:utf-8 -*-

import time
import os
import logging
import numpy
import pandas
import random
import glob
import cPickle as pickle
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.cross_validation import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV

#from xgboost import XGBClassifier, XGBRegressor

#from utils.load_data import load_data
#from utils.train_as_class import get_prob_argmax_label

#from utils.estimator import StackedEstimator, EnsembleEstimator
#from utils.round_estimator import RoundEstimator

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')
TEST_DATA = os.path.join(DATA_DIR, 'test/')


log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)


def main():

    list_file_path = ['submit_lasso_4.csv', 'submit_xgb_4.csv', 'submit_rf_4.csv']

    list_predict = []
    data = None
    for i in range(len(list_file_path)):
        df = pandas.read_csv(list_file_path[i], index_col='id')
        predict = df['Demanda_uni_equil'].values
        if data is not None:
            data = pandas.merge(data, df, left_index=True, right_index=True)
        else:
            data = df

    with open('mix_model.pkl', 'rb') as f:
        model = pickle.load(f)

    predict = model.predict(data.values)
    predict = numpy.where(predict < 0, 0, predict)
    df_ans = pandas.DataFrame(numpy.c_[data.index.values, predict],
                              columns=['id', 'Demanda_uni_equil'])
    df_ans['id'] = df_ans['id'].astype(int)
    df_ans.to_csv('submit_mix.csv', index=False)

if __name__ == '__main__':
    main()
