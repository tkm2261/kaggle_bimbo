# encoding: utf-8

import numpy
import pandas
import re
from collections import Counter
from gensim import corpora, matutils
from sklearn.feature_selection import f_regression

REGEX_NUMBER = re.compile('\d+')

REGEX_AMOUNT = re.compile('\d+')


if __name__ == '__main__':

    df1 = pandas.read_csv("../data/producto_tabla.csv", index_col='Producto_ID')
    df2 = pandas.read_csv("../data/product_demand.csv", index_col='Producto_ID')

    df = df1.merge(df2, right_index=True, left_index=True, how='inner')
    data = df['NombreProducto'].apply(lambda x: x.lower().split())
    data = data.apply(lambda x: [word for word in x if REGEX_AMOUNT.match(word) is None])

    dictionary = corpora.Dictionary(data.values)
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    for key, value in dictionary.items():
        print key, value
        break
    tmp = [dictionary.doc2bow(aaa) for aaa in data.values]
    bow = matutils.corpus2csc(tmp, dtype=numpy.int16).T

    result = f_regression(bow, df['pro_avg_demand'], center=False)
    result = pandas.DataFrame(numpy.array(result).T, columns=['F', 'p'])

    words = {dictionary[idx]: True for idx in result[result['p'] < 0.2].index}
    print len(words)
    for word in words:
        df1['wd_%s' % word] = data.apply(lambda x: word in x)

    def get_amount(row):
        reg = re.search('\s(\d+)[g|kg|ml|l]', row.lower())
        if reg is None:
            return 0
        else:
            return reg.group(1)

    df1['amount'] = df1['NombreProducto'].apply(lambda x: get_amount(x.lower()))

    df1 = df1.drop('NombreProducto', 1)
    df1.to_csv('product_master2.csv')
