import logging

import numpy as np

logger = logging.getLogger(__name__)

ITERS = 10
LAMBDA = .001
LOSS = 'hinge'
REGULARIZATION = 'none'
EXTEND_PWR_2 = True
EXTEND_LOG = True
EXTEND_SQRT = True
EXTEND_ABS = True

print('\n'+'#'*20)
print('\nIterations:\t{}'.format(ITERS))
print('lambda:\t\t{}'.format(LAMBDA))
print('Loss:\t\t{}'.format(LOSS))
print('Regularization: {}'.format(REGULARIZATION))
print('Features extension:')
print('X^2:\t\t{}'.format(EXTEND_PWR_2))
print('log(|X|+1):\t{}'.format(EXTEND_LOG))
print('sqrt(|X|):\t{}'.format(EXTEND_SQRT))
print('|X|:\t\t{}'.format(EXTEND_ABS))

print('')

np.random.seed(23)

class HingeLoss(object):
    """ Compute the value and gradient of the Hinge loss. """
    def value(self, X, y, w):
        return np.fmax(np.zeros(X.shape[0],), 1-y*w.dot(X.T))

    def grad(self, X, y, w):
        if y*w.T.dot(X) < 1:
            grad_w = -y * X
        else:
            grad_w = np.zeros(w.shape)
        return grad_w

class LogLoss(object):
    """ Compute the value and gradient of the Log loss. """
    def value(self, X, y, w):
        return np.log(1 + np.exp(-y*w.T.dot(X)))

    def grad(self, X, y, w):
        return -y*X / (1 + np.exp(y*w.T.dot(X)))


class SGDClassifier(object):
    loss_functions = {
        'hinge': HingeLoss(),
        'log': LogLoss()
    }

    def __init__(self, n_iterations, lambda_, loss='log', penalty='l2'):
        """ This class is a SGD Classifier which uses the given loss which can be:
        * hinge: Hinge Loss
        * log: logistic regression loss
        It also uses l2 or l1 penalty to the weight vector depending on the specification. """
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        assert loss in ('hinge', 'log'), 'Invalid given loss'
        self.loss = self.loss_functions[loss]
        penalty = penalty.lower()
        assert penalty in ('none', 'l2', 'l1')
        self.penalty = penalty

    def _apply_regularization(self, w):
        """ Compute the factor required to regularize the weight vector given this and the
        regularization desired. """
        if self.penalty == 'none':
            return 1.
        elif self.penalty == 'l1':
            return min(1.,
                       1. / (np.sqrt(self.lambda_) * np.linalg.norm(w, 1)))
        elif self.penalty == 'l2':
            return min(1.,
                       1. / (np.sqrt(self.lambda_) * np.linalg.norm(w, 2)))

    def fit(self, X, y):
        """ Fit the SGDClassifier finding the best weight vector for the given dataset (X, y). """
        nb_samples = X.shape[0]
        nb_features = X.shape[1]
        assert nb_samples == y.shape[0]

        # Initialize weight vector to zero
        w_ = np.zeros((nb_features,), dtype='float')

        for _ in range(self.n_iterations):
            # Shuffle data
            idx = np.random.permutation(nb_samples)
            X_ = X[idx,:]
            y_ = y[idx]

            for t in range(nb_samples):
                nhu = 1. / np.sqrt(nb_samples)
                w_ -= nhu * self.loss.grad(X_[t,:], y_[t], w_)
                # Scalar factor due to penalty
                w_ *= self._apply_regularization(w_)

        self.w_ = w_


def transform(X):

    # Extend the features with the some transformations of the given data
    X_ = X
    if EXTEND_LOG:
        X_ = np.hstack([X_, np.log(np.absolute(X) + 1)])
    if EXTEND_PWR_2:
        X_ = np.hstack([X_, X**2])
    if EXTEND_ABS:
        X_ = np.hstack([X_, np.absolute(X)])
    if EXTEND_SQRT:
        X_ = np.hstack([X_, np.sqrt(np.absolute(X))])

    # Make the data to have 0 mean and 1 std
    x_mean = X_.mean(axis=0, keepdims=True)
    x_std = X_.std(axis=0, keepdims=True)
    x = (X_-x_mean) / x_std

    # Add column of ones for the bias value of the weight vector
    x = np.hstack([x, np.ones((x.shape[0], 1))])

    return x

def read_value(value):
    """ Returns from the value given to the mapper as input, return the array representing the features 'X' and the label 'Y' """
    nb_samples = len(value)
    X = np.zeros((nb_samples, 400), dtype='float')
    Y = np.zeros((nb_samples, ), dtype='float')
    for i in range(nb_samples):
        line = value[i]
        v = line.strip().split(' ')
        Y[i] = float(v[0])
        X[i] = np.array([float(x) for x in v[1:]])
    return X, Y

def mapper(key, value):
    # key: None
    # value: one line of input file
    assert key is None, 'key is not None'

    X, Y = read_value(value)
    X = transform(X)
    svm = SGDClassifier(n_iterations=ITERS, lambda_=LAMBDA, loss=LOSS, penalty=REGULARIZATION)
    svm.fit(X, Y)

    logger.info('Finish mapper')
    yield 'w', svm.w_


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    # Stack all the weight vectors and compute the mean along all the mapper's solution
    w = np.vstack(values)
    w = np.mean(w, axis=0)
    yield w
