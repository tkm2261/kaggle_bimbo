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

    df1 = pandas.read_csv("../data/cliente_tabla.csv", index_col='Cliente_ID')
    df2 = pandas.read_csv("../data/cluente_trans_demand.csv.gz", index_col='Cliente_ID')

    df = df1.merge(df2, right_index=True, left_index=True, how='inner')
    data = df['NombreCliente'].apply(lambda x: x.lower().split())
    data = data.apply(lambda x: [word for word in x if REGEX_AMOUNT.match(word) is None])

    dictionary = corpora.Dictionary(data.values)
    dictionary.filter_extremes(no_below=10, no_above=0.3)
    for key, value in dictionary.items():
        print key, value
        break
    tmp = [dictionary.doc2bow(aaa) for aaa in data.values]
    bow = matutils.corpus2csc(tmp, dtype=numpy.int16).T

    result = f_regression(bow, df['cl_avg_demand'], center=False)
    result = pandas.DataFrame(numpy.array(result).T, columns=['F', 'p'])

    words = [dictionary[idx] for idx in result[result['p'] < 0.2].sort('p').index]

    print len(words)
    for word in words[:100]:
        print word, df1.columns.values
        df1['wd_%s' % word] = df1['NombreCliente'].apply(lambda x: word in x.decode('utf-8', 'ignore').lower())

    df1 = df1.drop('NombreCliente', 1)
    df1 = df1.ix[numpy.unique(df1.index.values), :]
    df1.to_csv('cliente_master2.csv')
