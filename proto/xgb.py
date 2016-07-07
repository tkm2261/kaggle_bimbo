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

from xgboost import XGBClassifier, XGBRegressor, DMatrix

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


def bimbo_obj(y_true, y_pred):
    error = numpy.log(y_pred + 1) - numpy.log(y_true.get_label() + 1)
    grad = error / (y_pred + 1)
    hess = (1 - error) / (y_pred + 1)**2
    return grad, hess


def bimbo_score_func(pred_y, y):
    score = numpy.sqrt(numpy.mean((numpy.log(pred_y + 1) - numpy.log(y + 1))**2))
    return score


def bimbo_score_func_xg(pred_y, y):
    score = numpy.sqrt(numpy.mean((numpy.log(pred_y + 1) - numpy.log(y.get_label() + 1))**2))
    return score


def bimbo_scoring(estimetor, X, y):
    """評価関数
    """

    pred_y = estimetor.predict(X)
    pred_y = numpy.where(pred_y < 0, 0, pred_y)
    return - bimbo_score_func(pred_y, y)


def main():

    list_file_path = sorted(glob.glob(os.path.join(DATA_DIR, 'train_all_join_4/*gz')))

    df = pandas.read_csv(list_file_path[0], compression='gzip')
    df = df.fillna(0)
    data = df[LIST_FEATURE_COLUMN_NAME].values
    target = df[TARGET_COLUMN_NAME].values

    model = XGBRegressor(seed=0)
    """
    params = {'max_depth': [20],
              'learning_rate': [0.1],
              'min_child_weight': [0.01],
              'subsample': [1],
              'colsample_bytree': [0.5],
              'reg_alpha': [0.01, 1, 10],
              'reg_lambda': [0.01, 1, 10],
              'colsample_bytree': [0.5],
              }
    cv = GridSearchCV(model, params, scoring=bimbo_scoring, n_jobs=1, refit=False, verbose=10)
    cv.fit(data, target)
    logger.info('cv best_params: %s' % cv.best_params_)
    """
    params = {'subsample': 1, 'learning_rate': 0.01, 'colsample_bytree': 0.5,
              'max_depth': 18, 'min_child_weight': 0.01, 'reg_alpha': 1., 'n_estimators': 500}

    logger.info('best_params: %s' % params)
    list_estimator = []
    for i in range(1, len(list_file_path)):
        logger.info('%s: %s' % (i, list_file_path[i]))
        model = XGBRegressor(seed=0)
        model.set_params(**params)
        test_df = pandas.read_csv(list_file_path[i], compression='gzip')
        test_df = test_df.fillna(0)
        test_data = test_df[LIST_FEATURE_COLUMN_NAME].values
        test_target = test_df[TARGET_COLUMN_NAME].values

        model.fit(data, target, eval_metric=bimbo_score_func_xg)
        list_estimator.append(model)

        if i == 1 or i == len(list_file_path) - 1:
            predict = numpy.mean([est.predict(data) for est in list_estimator], axis=0)
            predict = numpy.where(predict < 0, 0, predict)
            score = bimbo_score_func(predict, target)
            logger.info('INSAMPLE score: %s' % score)

            predict = numpy.mean([est.predict(test_data) for est in list_estimator], axis=0)
            predict = numpy.where(predict < 0, 0, predict)
            score = bimbo_score_func(predict, test_target)
            logger.info('score: %s' % score)

        # model.set_params(n_estimators=n_estimators)

        df = test_df
        data = test_data
        target = test_target

    with open('list_xgb_model_4.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)

if __name__ == '__main__':
    main()
