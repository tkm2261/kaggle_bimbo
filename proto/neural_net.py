#-*- coding:utf-8 -*-

import time
import os
import sys
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.grid_search import GridSearchCV

# from xgboost import XGBClassifier, XGBRegressor

import tensorflow as tf

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(APP_ROOT)

from poisson_sgd import SGDPoissonRegressor
# from utils.load_data import load_data
# from utils.train_as_class import get_prob_argmax_label

# from utils.estimator import StackedEstimator, EnsembleEstimator
# from utils.round_estimator import RoundEstimator

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_all_join000000000001.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'hogehoge')

TARGET_COLUMN_NAME = 't_t_target'
from feature_3 import LIST_FEATURE_COLUMN_NAME


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

    list_file_path = sorted(glob.glob(os.path.join(DATA_DIR, 'train_join_all_3/*gz')))

    df = pandas.read_csv(list_file_path[0], compression='gzip')
    df = df.fillna(0)
    data = df[LIST_FEATURE_COLUMN_NAME].values
    target = df[TARGET_COLUMN_NAME].values

    df_non_zero = df[df[TARGET_COLUMN_NAME] > 0]
    non_zero_data = df_non_zero[LIST_FEATURE_COLUMN_NAME].values
    non_zero_target = df_non_zero[TARGET_COLUMN_NAME].values

    zero_target = numpy.where(target > 0, 1, 0)

    sess = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, data.shape[1]])
    W = tf.Variable(tf.zeros([data.shape[1], 2000]))
    b = tf.Variable(tf.zeros([2000]))
    h1 = tf.nn.relu(tf.matmul(x, W) + b)

    W2 = tf.Variable(tf.zeros([2000, 1000]))
    b2 = tf.Variable(tf.zeros([1000]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.Variable(tf.zeros([1000, 500]))
    b3 = tf.Variable(tf.zeros([500]))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

    W4 = tf.Variable(tf.zeros([500, 200]))
    b4 = tf.Variable(tf.zeros([200]))
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)

    W5 = tf.Variable(tf.zeros([200, 100]))
    b5 = tf.Variable(tf.zeros([100]))
    h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)

    W6 = tf.Variable(tf.zeros([100, 10]))
    b6 = tf.Variable(tf.zeros([10]))
    h6 = tf.nn.relu(tf.matmul(h5, W6) + b6)

    W7 = tf.Variable(tf.zeros([10, 1]))
    b7 = tf.Variable(tf.zeros([1]))
    h7 = tf.matmul(h6, W7) + b7
    y = h7

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])
    square_error = tf.sqrt(tf.reduce_mean(tf.square(tf.log(y_ + 1) - tf.log(y + 1))))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(square_error)
    accuracy = tf.reduce_mean(tf.square(tf.log(y_ + 1) - tf.log(y + 1)))
    # Train
    tf.initialize_all_variables().run()
    batch_size = 1000

    for i in range(1, len(list_file_path)):
        logger.info('%s: %s' % (i, list_file_path[i]))

        test_df = pandas.read_csv(list_file_path[i], compression='gzip')
        test_df = test_df.fillna(0)
        test_data = test_df[LIST_FEATURE_COLUMN_NAME].values
        test_target = test_df[TARGET_COLUMN_NAME].values
        test_zero_target = numpy.where(test_target > 0, 1, 0)

        test_df_non_zero = test_df[test_df[TARGET_COLUMN_NAME] > 0]
        test_non_zero_data = test_df_non_zero[LIST_FEATURE_COLUMN_NAME].values
        test_non_zero_target = test_df_non_zero[TARGET_COLUMN_NAME].values

        # model.fit(non_zero_data, non_zero_target)
        for _ in range(10):
            for i in range(int(data.shape[0] / batch_size)):
                start = batch_size * i
                end = start + batch_size
                if end > data.shape[0]:
                    end = data.shape[0]
                batch_xs = data[start: end]
                batch_ys = target[start: end]
                train_step.run({x: batch_xs, y_: batch_ys.reshape(-1, 1)})
                # Test trained model

            score = numpy.sqrt(accuracy.eval({x: data, y_: target.reshape(-1, 1)}))
            logger.info('INSAMPLE score: %s' % score)

        score = numpy.sqrt(accuracy.eval({x: test_data, y_: test_target.reshape(-1, 1)}))
        logger.info('score: %s' % score)

        df = test_df
        data = test_data
        target = test_target
        zero_target = test_zero_target
        df_non_zero = test_df_non_zero
        non_zero_data = test_non_zero_data
        non_zero_target = test_non_zero_target

if __name__ == '__main__':
    main()
