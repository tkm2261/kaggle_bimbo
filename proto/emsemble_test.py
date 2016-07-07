# -*- coding:utf-8 -*-

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

from sklearn.linear_model import LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.cross_validation import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV

from xgboost import XGBClassifier, XGBRegressor

#from utils.load_data import load_data
#from utils.train_as_class import get_prob_argmax_label

#from utils.estimator import StackedEstimator, EnsembleEstimator
#from utils.round_estimator import RoundEstimator

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_all_join000000000001.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'hogehoge')

TARGET_COLUMN_NAME = 't_t_target'
from feature_4 import LIST_FEATURE_COLUMN_NAME
# best_params: {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree':
# 0.5, 'max_depth': 10, 'min_child_weight': 0.01}

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)


def bimbo_score_func(pred_y, y):
    score = numpy.sqrt(numpy.mean((numpy.log(pred_y + 1) - numpy.log(y + 1))**2))
    return score


def bimbo_scoring(estimetor, X, y):
    """評価関数
    """

    pred_y = estimetor.predict(X)
    pred_y = numpy.where(pred_y < 0, 0, pred_y)
    return - bimbo_score_func(pred_y, y)


def main():

    list_file_path = sorted(glob.glob(os.path.join(DATA_DIR, 'train_join_all_4/*gz')))

    df = pandas.read_csv(list_file_path[-1], compression='gzip')
    df = df.fillna(0)
    data = df[LIST_FEATURE_COLUMN_NAME].values
    target = df[TARGET_COLUMN_NAME].values

    with open('lasso_model_3.pkl', 'rb') as f:
        lasso_model = pickle.load(f)

    with open('list_xgb_model_3.pkl', 'rb') as f:
        list_xgb_model = pickle.load(f)

    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    list_predict = []

    predict = lasso_model.predict(data)
    predict = numpy.where(predict < 0, 0, predict)
    score = bimbo_score_func(predict, target)
    logger.info('lasso score: %s' % score)
    list_predict.append(predict)

    predict = numpy.mean([est.predict(data) for est in list_xgb_model], axis=0)
    predict = numpy.where(predict < 0, 0, predict)
    score = bimbo_score_func(predict, target)
    logger.info('xgb score: %s' % score)
    list_predict.append(predict)

    predict = rf_model.predict(data)
    predict = numpy.where(predict < 0, 0, predict)
    score = bimbo_score_func(predict, target)
    logger.info('rf score: %s' % score)
    list_predict.append(predict)

    data = numpy.array(list_predict).T
    model = LinearRegression()

    score = numpy.mean(cross_val_score(model, data, target, scoring=bimbo_scoring))
    logger.info('mix score: %s' % score)
    model.fit(data, target)

    with open('mix_model.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
if __name__ == '__main__':
    main()
