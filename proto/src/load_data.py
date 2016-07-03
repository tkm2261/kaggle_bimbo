#-*- coding:utf-8 -*-

import numpy
import pandas
from sklearn.preprocessing import OneHotEncoder
from logging import getLogger

logger = getLogger(__name__)

class FeatureData:
    """データ格納クラス

    ;param numpy.ndarray data; 特徴量
    :param numpy.ndarray target: ラベル 1-of-K符号化済み
    :param sklearn.preprocessing.OneHotEncoder target_one_hot_encoder: ラベル1-of-K符号化用クラス
    """
    
    def __init__(self,
                 data,
                 target,
                 target_one_hot_encoder):
        self.data = data
        self.target = target
        self.target_one_hot_encoder = target_one_hot_encoder

def load_data(file_path,
              target_column,
              list_feature_columns,
              target_one_hot_encoder=None,
              is_test_data=False):
    """データ読込関数

    :param str file_path: データCSVパス
    :param str target_column: ラベルカラム名
    :param list list_feature_columns: 特徴量カラム名のリスト
    :param sklearn.preprocessing.OneHotEncoder target_one_hot_encoder: ラベル1-of-K符号化用クラス
    :param is_test_data: テストデータかどうか
    """
    logger.debug('enter')
    df = pandas.read_csv(file_path, dtype=numpy.float)

    data = df[list_feature_columns].values

    if not is_test_data:
        target_one_hot_encoder = OneHotEncoder(sparse=False, dtype=numpy.bool)
        target_one_hot_encoder.fit(df[target_column].values.reshape(-1, 1))
        logger.info('active_features: {}'.format(target_one_hot_encoder.active_features_))
        logger.info('feature_indices: {}'.format(target_one_hot_encoder.feature_indices_))
        logger.info('n_values: {}'.format(target_one_hot_encoder.n_values_))

    if is_test_data:
        target_data = None
    else:
        target_data = target_one_hot_encoder.transform(df[target_column].values.reshape(-1, 1))
        logger.debug('target distribution: {}'.format(df.groupby(target_column)[target_column].count() / df.shape[0]))
        logger.debug('target feature distribution: {}'.format(df.groupby(target_column).mean()))

    logger.debug('exit')

    return FeatureData(data, target_data, target_one_hot_encoder)
