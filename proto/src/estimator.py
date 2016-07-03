#-*- coding:utf-8 -*-

import pandas
import numpy
from collections import Counter
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from logging import getLogger

logger = getLogger(__name__)

class StackedEstimator:
    def __init__(self,
                 list_estimetor,
                 cross_validation_pertition,
                 tunning_scoring=None,
                 random_state=111):
    
        self.list_estimetor = list_estimetor
        self.cross_validation_pertition = cross_validation_pertition
        self.random_state = random_state

        self.positive_ratio = None
        
    def fit(self, X, y, sample_weight=None):
        logger.debug('enter')

        self.positive_ratio = y.sum() / max(y.shape)
        for i, estimetor in enumerate(self.list_estimetor):
            logger.info('stack; {}'.format(i+1))
            if i > 0:
                X = numpy.array(new_X)

            new_X = numpy.zeros((X.shape[0], estimetor.n_estimator))
            all_predicts = numpy.zeros(y.shape)
            all_predicts_proba = numpy.zeros(y.shape)
            
            for j, (train_index, test_index) in enumerate(self.cross_validation_pertition):
                logger.debug('global cv; {}'.format(j+1))
                train_data = X[train_index]
                train_target = y[train_index]
                test_data = X[test_index]
                test_target = y[test_index]

                estimetor.fit(train_data, train_target)
                predicts_proba = estimetor.predict_all_estimater(test_data)
                new_X[test_index] = predicts_proba.values
                all_predicts[test_index] = estimetor.predict(test_data)

            estimetor.fit(X, y)
            logger.info('stack; {} Estimator {}'.format(i+1, predicts_proba.columns))
            corr_matrix = numpy.corrcoef(new_X.T)
            for s in range(len(predicts_proba.columns)):
                for t in range(len(predicts_proba.columns)):
                    try:
                        logger.info('stack; {} Correlations Values:({}, {}) are {}'.format(i+1,
                                                                                           predicts_proba.columns[s],
                                                                                           predicts_proba.columns[t],
                                                                                           corr_matrix[s, t]))
                    except IndexError:
                        pass
            logger.info('stack; {} Accuracy score is {}'.format(i+1, accuracy_score(y, all_predicts)))
            
        logger.debug('exit')
        return self
    
    def predict(self, X):
        logger.debug('enter')
        for i, estimetor in enumerate(self.list_estimetor):
            logger.info('stack; {}'.format(i+1))
            if i > 0:
                X = numpy.array(new_X)
            new_X = numpy.array(estimetor.predict_all_estimater(X))

        df = pandas.DataFrame(new_X).apply(lambda row: Counter(row).most_common(1)[0][0], axis=1)
        logger.debug('exit')
        return df.values


    
class EnsembleEstimator:

    def __init__(self,
                 map_estimator,
                 map_estimator_parameter,
                 tunning_scoring=None,
                 n_folds=5,
                 random_state=111,
                 n_jobs=-1):

        self.n_estimator = len(map_estimator)
        self.map_estimator = map_estimator
        self.map_estimator_parameter = map_estimator_parameter
        self.tunning_scoring = tunning_scoring
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.positive_ratio = None
        self.cross_validation_pertition = None
        
    def fit(self, X, y, sample_weight=None):
        logger.debug('enter')

        self.positive_ratio = y.sum() / max(y.shape)
        self.cross_validation_pertition = StratifiedKFold(y,
                                                          n_folds=self.n_folds,
                                                          shuffle=True,
                                                          random_state=self.random_state)

        for estimator_name in sorted(self.map_estimator.keys()):
            logger.debug('start learing {}'.format(estimator_name))
            estimator = self.map_estimator[estimator_name]
            estimator_parameter = self.map_estimator_parameter[estimator_name]
            grid_search = GridSearchCV(estimator,
                                       estimator_parameter,
                                       scoring=self.tunning_scoring,
                                       cv=self.cross_validation_pertition,
                                       n_jobs=self.n_jobs)

            grid_search.fit(X, y)
            self.map_estimator[estimator_name] = grid_search.best_estimator_
            logger.debug('end learing. best score is {}'.format(grid_search.best_score_))
            logger.debug('best param is {}'.format(grid_search.best_params_))

        logger.debug('exit')
        return self
    
    def predict_all_estimater(self, X):
        logger.debug('enter')
        df = pandas.DataFrame()
        for estimator_name, estimator in sorted(self.map_estimator.items()):
            df[estimator_name] = estimator.predict(X)

        df = numpy.around(df).astype(int)
        logger.debug('exit')
        return df

    def get_estimator_correlation_value(self, X):
        logger.debug('enter')
        predicts_proba = self.predict_all_estimater(X)
        
        return predicts_proba.columns, predicts_proba.corr()

    
    def predict(self, X):
        logger.debug('enter')
        df = self.predict_all_estimater(X)
        df = df.apply(lambda row: Counter(row).most_common(1)[0][0], axis=1)
        logger.debug('exit')
        return df.values
