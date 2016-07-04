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
TEST_DATA = os.path.join(DATA_DIR, 'test_all_join_3/')

TARGET_COLUMN_NAME = 't_t_target'
from feature_3 import LIST_FEATURE_COLUMN_NAME

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)


def bimbo_scoring(estimetor, X, y):
    """評価関数
    """

    pred_y = estimetor.predict(X)
    pred_y = numpy.where(pred_y < 0, 0, pred_y)
    score = numpy.sqrt(numpy.mean((numpy.log(pred_y + 1) - numpy.log(y + 1))**2))
    return score


def main():

    list_file_path = glob.glob(os.path.join(TEST_DATA, '*gz'))

    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    df_ans = pandas.DataFrame()
    for i in range(len(list_file_path)):
        logger.info('%s: %s' % (i, list_file_path[i]))
        df = pandas.read_csv(list_file_path[i], compression='gzip')
        logger.info('end load')
        df = df.fillna(0)
        data = df[LIST_FEATURE_COLUMN_NAME].values
        predict = model.predict(data)
        predict = numpy.where(predict < 0, 0, predict)
        logger.info('end predict')
        ans = pandas.DataFrame(df['t_id'])
        ans.columns = ['id']
        ans['Demanda_uni_equil'] = predict
        df_ans = df_ans.append(ans)

    df_ans.to_csv('submit_rf.csv', index=False)

if __name__ == '__main__':
    main()
