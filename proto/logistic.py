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
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV, ParameterGrid

#from xgboost import XGBClassifier, XGBRegressor

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(APP_ROOT)

from poisson_sgd import SGDPoissonRegressor
#from utils.load_data import load_data
#from utils.train_as_class import get_prob_argmax_label

#from utils.estimator import StackedEstimator, EnsembleEstimator
#from utils.round_estimator import RoundEstimator

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_all_join000000000001.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'hogehoge')

TARGET_COLUMN_NAME = 't_t_target'
LIST_FEATURE_COLUMN_NAME = ['t_t_Agencia_ID', 't_t_Canal_ID', 't_t_Ruta_SAK', 't_t_Cliente_ID', 't_t_Producto_ID',
                            't_t_Venta_uni_hoy', 't_t_Venta_hoy', 't_t_Dev_uni_proxima', 't_t_Dev_proxima',
                            't_t_Demanda_uni_equil', 't_a_lat', 't_a_lon', 't_p_amount', 't_p_mk_bim',
                            't_p_mk_mla', 't_p_mk_tr', 't_p_mk_lar', 't_p_mk_gbi', 't_p_mk_won', 't_p_mk_dh', 't_p_mk_lon',
                            't_p_mk_san', 't_p_mk_mr', 't_p_mk_oro', 't_p_mk_cc', 't_p_mk_sl', 't_p_mk_bar', 't_p_mk_sua',
                            't_p_mk_ric', 't_p_mk_mp', 't_p_mk_sun', 't_p_mk_jmx', 't_p_mk_skd', 't_p_wd_mta', 't_p_wd_tab',
                            't_p_wd_cu', 't_p_wd_pan', 't_p_wd_mtb', 't_p_wd_sp', 't_p_wd_prom', 't_p_wd_fresa', 't_p_wd_duo',
                            't_p_wd_tubo', 't_p_wd_vainilla', 't_p_wd_deliciosas', 't_p_wd_blanco', 't_p_wd_chocolate', 't_p_wd_tnb',
                            't_p_wd_tira', 't_p_wd_cj', 't_p_wd_gansito', 't_p_wd_suavicremas', 't_p_wd_galleta', 't_p_wd_nuez',
                            't_p_wd_multigrano', 't_p_wd_pina', 't_p_wd_tortilla', 't_p_wd_barritas', 't_p_wd_tostada', 't_p_wd_frut',
                            't_p_wd_bran', 't_p_wd_lata', 't_p_wd_principe', 't_p_wd_me', 't_p_wd_mantecadas', 't_p_wd_mini', 't_p_wd_kc',
                            't_p_wd_triki', 't_p_wd_tostado', 't_p_wd_mg', 't_p_wd_integral', 't_p_wd_roles', 't_p_wd_bollos',
                            't_p_wd_bimbollos', 't_p_wd_tortillinas', 't_p_wd_medias', 't_p_wd_canelitas', 't_p_wd_fibra',
                            't_p_wd_noches', 't_p_wd_barra', 't_p_wd_con', 't_p_wd_de', 't_p_wd_cjm', 't_p_wd_hna', 't_p_wd_mi',
                            't_p_wd_super', 't_p_wd_tartinas', 't_p_wd_cr2', 't_p_wd_trakes', 't_p_wd_surtido', 't_p_wd_mas',
                            't_p_wd_canela', 't_p_wd_avena', 't_p_wd_nito', 't_p_wd_plativolos', 't_p_wd_choco', 't_p_wd_chocochispas', 't_p_wd_sandwich',
                            't_p_wd_doble', 't_p_wd_pack', 't_p_wd_submarinos', 't_p_wd_salmas', 't_p_wd_silueta', 't_p_wd_hot', 't_p_wd_clasico', 't_p_wd_linaza',
                            't_p_wd_maiz', 't_p_wd_donas', 't_p_wd_lors', 't_p_wd_bco', 't_p_wd_harina', 't_p_wd_cr1', 't_p_wd_panera', 't_p_wd_polvorones',
                            't_p_wd_wonder', 't_p_wd_azucar', 't_p_wd_besos', 't_p_wd_choc', 't_p_wd_ondulada', 't_p_wd_jamon', 't_p_wd_mm', 't_p_wd_bk',
                            't_p_wd_extra', 't_p_wd_molido', 't_p_wd_barrita', 't_p_wd_rocko', 't_p_wd_galletas', 't_p_wd_rico', 't_p_wd_chochitos',
                            't_p_wd_ajonjoli', 't_p_wd_canapinas', 't_p_wd_dif', 't_p_wd_fs', 'at_ag_cnt', 'at_ag_max_venta_uni', 'at_ag_min_venta_uni',
                            'at_ag_avg_venta_uni', 'at_ag_max_venta_hoy', 'at_ag_min_venta_hoy', 'at_ag_avg_venta_hoy', 'at_ag_max_dev_uni', 'at_ag_min_dev_uni',
                            'at_ag_avg_dev_uni', 'at_ag_max_dev_hoy', 'at_ag_min_dev_hoy', 'at_ag_avg_dev_hoy', 'at_ag_max_demand', 'at_ag_min_demand',
                            'at_ag_avg_demand', 'ch_ch_cnt', 'ch_ch_max_venta_uni', 'ch_ch_min_venta_uni', 'ch_ch_avg_venta_uni', 'ch_ch_max_venta_hoy',
                            'ch_ch_min_venta_hoy', 'ch_ch_avg_venta_hoy', 'ch_ch_max_dev_uni', 'ch_ch_min_dev_uni', 'ch_ch_avg_dev_uni', 'ch_ch_max_dev_hoy',
                            'ch_ch_min_dev_hoy', 'ch_ch_avg_dev_hoy', 'ch_ch_max_demand', 'ch_ch_min_demand', 'ch_ch_avg_demand', 'cl_cl_cnt', 'cl_cl_max_venta_uni',
                            'cl_cl_min_venta_uni', 'cl_cl_avg_venta_uni', 'cl_cl_max_venta_hoy', 'cl_cl_min_venta_hoy', 'cl_cl_avg_venta_hoy', 'cl_cl_max_dev_uni',
                            'cl_cl_min_dev_uni', 'cl_cl_avg_dev_uni', 'cl_cl_max_dev_hoy', 'cl_cl_min_dev_hoy', 'cl_cl_avg_dev_hoy', 'cl_cl_max_demand', 'cl_cl_min_demand', 'cl_cl_avg_demand']


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
    return bimbo_score_func(pred_y, y)

def main():


    list_file_path = glob.glob(os.path.join(DATA_DIR, '*gz'))

    df = pandas.read_csv(list_file_path[0], compression='gzip')
    df = df.fillna(0)
    data = df[LIST_FEATURE_COLUMN_NAME].values
    target = df[TARGET_COLUMN_NAME].values

    df_non_zero = df[df[TARGET_COLUMN_NAME] > 0]
    non_zero_data = df_non_zero[LIST_FEATURE_COLUMN_NAME].values
    non_zero_target = df_non_zero[TARGET_COLUMN_NAME].values

    zero_target = numpy.where(target > 0, 1, 0)

    test_df = pandas.read_csv(list_file_path[1], compression='gzip')
    test_df = test_df.fillna(0)
    test_data = test_df[LIST_FEATURE_COLUMN_NAME].values
    test_target = test_df[TARGET_COLUMN_NAME].values
    test_zero_target = numpy.where(test_target > 0, 1, 0)

    test_df_non_zero = test_df[test_df[TARGET_COLUMN_NAME] > 0]
    test_non_zero_data = test_df_non_zero[LIST_FEATURE_COLUMN_NAME].values
    test_non_zero_target = test_df_non_zero[TARGET_COLUMN_NAME].values


    
    n_estimators = 30
    """
    zero_model = RandomForestClassifier(n_estimators=n_estimators,
                                        min_samples_leaf=10,
                                        random_state=0,
                                        warm_start=True,
                                        n_jobs=-1)
    """
    """
    model = SGDRegressor(loss='huber',
                         penalty='l1',
                         average=100,
                         warm_start=True)
    """
    model = ElasticNet(random_state=0)
    all_params = {'alpha': [0.01, 0.1, 1],
                  'l1_ratio': [0.3, 0.5, 0.7]}

    max_score = - 1.0E10
    max_params = None
    logger.info('cv start')    
    _, train_idx = list(KFold(data.shape[0], 100, shuffle=True))[0]
    for params in ParameterGrid(all_params):
        model.set_params(**params)
        model.fit(data[train_idx], numpy.log(target[train_idx] + 1))
        predict = model.predict(test_data)
        predict = numpy.exp(predict) - 1
        score = bimbo_score_func(predict, test_target)
        logger.info('params: %s'%(params))
        logger.info('score: %s, best_score: %s'%(score, max_score))
        if score > max_score:
            max_params = params
            max_score = score
            
    model.set_params(**max_params)
    model.set_params(warm_start=True)

    for i in range(1, len(list_file_path)):
        logger.info('%s: %s'%(i, list_file_path[i]))

        test_df = pandas.read_csv(list_file_path[i], compression='gzip')
        test_df = test_df.fillna(0)
        test_data = test_df[LIST_FEATURE_COLUMN_NAME].values
        test_target = test_df[TARGET_COLUMN_NAME].values
        test_zero_target = numpy.where(test_target > 0, 1, 0)

        test_df_non_zero = test_df[test_df[TARGET_COLUMN_NAME] > 0]
        test_non_zero_data = test_df_non_zero[LIST_FEATURE_COLUMN_NAME].values
        test_non_zero_target = test_df_non_zero[TARGET_COLUMN_NAME].values

        model.fit(data, numpy.log(target + 1))
        #zero_model.fit(data, zero_target)

        #zero_predict = zero_model.predict(data)
        predict = model.predict(data)
        predict = numpy.exp(predict) - 1
        #predict = zero_predict * predict

        score = bimbo_score_func(predict, target)
        logger.info('INSAMPLE score: %s'%score)
        #zero_predict = zero_model.predict(test_data)
        predict = model.predict(test_data)
        predict = numpy.exp(predict) - 1
        #predict = zero_predict * predict

        score = bimbo_score_func(predict, test_target)
        logger.info('score: %s'%score)

        #zero_model.set_params(n_estimators=n_estimators)

        df = test_df
        data = test_data
        target = test_target
        zero_target = test_zero_target
        df_non_zero = test_df_non_zero
        non_zero_data = test_non_zero_data
        non_zero_target = test_non_zero_target

        
    with open('sgd_model.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
 
    """
    df = pandas.read_csv(list_file_path[0], compression='gzip')
    df = df.fillna(0)
    data = df[LIST_FEATURE_COLUMN_NAME].values.astype(numpy.float)
    target = df[TARGET_COLUMN_NAME].values.astype(numpy.float)
    n_estimators = 10
    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                      min_samples_leaf=10,
                                      random_state=0,
                                      warm_start=False)



    for i in range(1, len(list_file_path)):
        logger.info('%s: %s'%(i, list_file_path[i]))

        test_df = pandas.read_csv(list_file_path[i], compression='gzip')
        test_df = test_df.fillna(0)
        test_data = numpy.asarray(test_df[LIST_FEATURE_COLUMN_NAME].values, order='C')
        test_target = numpy.asarray(test_df[TARGET_COLUMN_NAME].values, order='C')
        
        model.fit(data, target)
        score = bimbo_scoring(model, data, target)
        logger.info('INSAMPLE score: %s'%score)
        score = bimbo_scoring(model, test_data, test_target)
        logger.info('score: %s'%score)

        n_estimators += 5
        model.set_params(n_estimators=n_estimators)

        df = test_df
        data = test_data
        target = test_target

    with open('gbt_model.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
        
    """

if __name__ == '__main__':
    main()
