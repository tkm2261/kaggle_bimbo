# encoding: utf-8
import numpy
from sklearn.preprocessing import StandardScaler
from logging import getLogger

logger = getLogger(__name__)

class SGDPoissonRegressor(object):

    def __init__(self,
                 alpha=0.1,
                 n_iter=5,
                 random_state=0,
                 power_t=0.25,
                 eta0=0.001):
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state
        numpy.random.seed(self.random_state)

        self.power_t = power_t
        self.eta0 = eta0

        self.standard_scalar = StandardScaler()
        self.coef_ = None
        self.intercept_ = None

    def fit(self, _X, y):
        if self.coef_ is None or self.intercept_ is None:
            self.coef_ = numpy.ones(_X.shape[1]) * 1.0E-03
            self.intercept_ = 1.
            self.standard_scalar.fit(_X)

        X = self.standard_scalar.transform(_X)
        """
        from scipy.optimize import minimize
        
        def func(x):
            coef_ = x[:-1]
            intercept_ = x[-1]
            val1 = numpy.dot(X, coef_) + intercept_
            val2 = numpy.exp(val1)
            like = - numpy.sum(y * val1 - val2)
            return like
            
        res = minimize(func, numpy.r_[self.coef_, self.intercept_])
        print "AAA", res.success, res.message
        self.coef_ = res.x[:-1]
        self.intercept_ = res.x[-1]
        """
        for iter in range(self.n_iter):
            logger.debug('iter: %s'%(iter+1))
            self.partial_fit(X, y, iter + 1)

    def predict(self, _X):
        X = self.standard_scalar.transform(_X)
        return self._input_func(X)

    def partial_fit(self, X, y, t=1):

        learning_rate = self._get_learning_rate(t)
        #total_coef_loss = numpy.zeros(self.coef_.shape)
        #total_intercept_loss = 0.
        for i in numpy.arange(X.shape[0]):
            try:
                coef_loss, intercept_loss = self._get_grad(X[i], y[i])
            except Exception:
                break
            #total_coef_loss = total_coef_loss + coef_loss
            #total_intercept_loss = total_intercept_loss + intercept_loss

            self.coef_ = self.coef_ + (learning_rate / X.shape[0]) * coef_loss
            self.intercept_ = self.intercept_ + (learning_rate / X.shape[0]) * intercept_loss

            if i % 10000 == 0:
                print self.intercept_
                
    def _get_grad(self, x, y):
        activate = self._input_func(x)
        if activate != activate:
            raise Exception('aaa')
        loss = y - activate
        coef_loss = loss * x
        coef_loss = numpy.where(coef_loss > 0, coef_loss - self.alpha, coef_loss + self.alpha)
        coef_loss = numpy.where((coef_loss < self.alpha)&(coef_loss > - self.alpha), 0, coef_loss)
        intercept_loss = loss

        return coef_loss, intercept_loss

    def _input_func(self, X):
        return numpy.exp(numpy.dot(X, self.coef_)) + self.intercept_

    def _get_learning_rate(self, t):
        return self.eta0 / numpy.power(t, self.power_t)

if __name__ == '__main__':
    model = SGDPoissonRegressor()
    X = [[0, 1, 21],
         [0, 1, 30],
         [0, 1, 37],
         [0, 2, 46],
         [1, 1, 24],
         [1, 1, 56],
         [1, 1, 58],
         [1, 2, 24],
         [1, 2, 38],
         [1, 2, 58],
         [1, 3, 26],
         [1, 3, 41],
         [0, 1, 23],
         [0, 1, 43],
         [0, 1, 47],
         [0, 2, 35],
         [0, 2, 41],
         [0, 2, 45],
         [0, 2, 53],
         [0, 3, 40],
         [1, 1, 22],
         [1, 1, 39],
         [1, 1, 52],
         [1, 2, 23],
         [1, 2, 28],
         [1, 2, 32],
         [1, 2, 43],
         [1, 3, 24],
         [1, 3, 27],
         [1, 3, 42],
         [0, 1, 20],
         [0, 1, 44],
         [0, 2, 35],
         [0, 2, 37],
         [0, 3, 41],
         [0, 3, 55],
         [0, 3, 51],
         [0, 3, 36],
         [1, 1, 34],
         [1, 2, 42],
         [1, 2, 51],
         [1, 3, 21],
         [1, 3, 35],
         [1, 3, 36]]
    
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    X = numpy.array(X)
    y = numpy.array(y)
    for _ in range(10):
        model.fit(X, y)
        #print model.coef_, model.intercept_
        pred = model.predict(X)
        print pred#numpy.sqrt(numpy.mean((y -pred) **2))

        val1 = numpy.dot(X, model.coef_) + model.intercept_
        val2 = numpy.exp(val1)
        like = - numpy.sum(y * val1 - val2)
        print like
