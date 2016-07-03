#-*- coding:utf-8 -*-

import numpy

def get_prob_argmax_label(label_prob_matrix, list_label_names):
    """ラベル確率が最も高いものに分類する関数

    :param label_prob_matrix: [データ数, ラベルの確率]　行列
    :param list_label_names: ラベル（列）の名前
    """
        
    return [list_label_names[numpy.argsort(label_prob_matrix[i, :])[0]]
            for i in range(label_prob_matrix.shape[0])]
    

