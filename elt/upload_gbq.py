# encoding: utf-8

import pandas
import re
from collections import Counter

REGEX_NUMBER = re.compile('\d+')

REGEX_AMOUNT = re.compile('\d+[g|kg|ml|l]')


if __name__ == '__main__':

    df = pandas.read_csv('cliente_master2.csv', index_col=['Cliente_ID'])
    print df.shape
    df = df.fillna(False)
    pandas.io.gbq.to_gbq(df, 'train.cliente_master3', '435429141811', chunksize=10000)
    """

    df = pandas.read_csv('product_master2.csv', index_col=['Producto_ID'])
    df = df.fillna(False)
    print df.shape
    pandas.io.gbq.to_gbq(df, 'train.product_master3', '435429141811')
    """
