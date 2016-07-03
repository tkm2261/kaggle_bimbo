# encoding: utf-8

import pandas
import re
from collections import Counter

REGEX_NUMBER = re.compile('\d+')

REGEX_AMOUNT = re.compile('\d+[g|kg|ml|l]')


def make_maker_dict(data, n_top=20):
    counter = Counter(data)
    return counter.most_common(n_top)


if __name__ == '__main__':

    data = pandas.read_csv("producto_tabla.csv")
    maker = data['NombreProducto'].apply(lambda x: x.split()[-2])

    list_maker = [row[0].lower() for row in make_maker_dict(maker.values)]

    words = data['NombreProducto'].apply(lambda x: " ".join(x.split()[:-2]))
    words = dict(Counter(" ".join(words).lower().split()))
    words = {word: count for word, count in words.items() if len(word) > 1 and REGEX_NUMBER.match(word) is None}

    df_words = pandas.Series(words)
    df_words.sort(ascending=False)
    # for idx in words.iloc[:100, ].index.values:
    #    print idx, words[idx]
    df = pandas.DataFrame(data['Producto_ID'])

    words = data['NombreProducto'].apply(lambda x: x.lower().split())

    def get_amount(row):
        reg = re.search('\s(\d+)[g|kg|ml|l]', row.lower())
        if reg is None:
            return 0
        else:
            return reg.group(1)

    df['amount'] = data['NombreProducto'].apply(lambda x: get_amount(x))
    for maker in list_maker:
        df['mk_%s' % maker] = words.apply(lambda x: x[-2] == maker)

    for idx in df_words.iloc[:100, ].index.values:
        word = idx
        df['wd_%s' % word] = words.apply(lambda x: word in x[:-2])

    print df
    df.to_csv('product_master.csv', index=False)
