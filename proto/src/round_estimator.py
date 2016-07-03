#-*- coding:utf-8 -*-

import pandas
import numpy
from logging import getLogger

logger = getLogger(__name__)

MIN_QUALITY = 3
MAX_QUALITY = 9

class RoundEstimator:
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        
        try:
            self.estimetor = args[0]
        except IndexError:
            self.estimetor = kwargs.pop('estimetor')
            
        self.estimetor.set_params(**kwargs)
        
    def fit(self, X, y):
        self.estimetor.fit(X, y)
        return self
    
    def predict_proba(self, X):
        result = self.estimetor.predict_proba(X)
        return result

    def predict(self, X):
        pred_y = self.estimetor.predict(X)
        pred_y = numpy.around(pred_y)
        pred_y = numpy.where(pred_y < MIN_QUALITY, MIN_QUALITY, pred_y)
        pred_y = numpy.where(pred_y > MAX_QUALITY, MAX_QUALITY, pred_y)
        return pred_y

    def decision_function(self, X):
        result = self.estimetor.decision_function(X)
        return result
        
    
    def get_params(self, deep=True):
        map_param = self.estimetor.get_params(deep)
        map_param.update({'estimetor': self.estimetor})

        return map_param
        
    def set_params(self, **kwargs):
        if 'estimetor' in kwargs:
            self.estimetor = kwargs.pop('estimetor')
        self.estimetor.set_params(**kwargs)
        return self
