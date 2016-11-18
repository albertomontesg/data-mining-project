import logging

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)

ITERS = 10
LAMBDA = .000001
LOSS = 'hinge'
REGULARIZATION = 'l2'
AVERAGING = True

RBF = False
GAMMA = .001
RBF_SPACE = 10000

print('\n'+'#'*20)
print('\nIterations: {}'.format(ITERS))
print('lambda: {}'.format(LAMBDA))
print('Loss: {}'.format(LOSS))
print('Regularization: {}'.format(REGULARIZATION))
print('Averaging: {}'.format(AVERAGING))
if RBF:
    print('Gamma: {}'.format(GAMMA))
    print('RBF space: {}'.format(RBF_SPACE))
print('')

np.random.seed(23)

class HingeLoss(object):
    def value(self, X, y, w):
        return np.fmax(np.zeros(X.shape[0],), 1-y*w.dot(X.T))

    def grad(self, X, y, w):
        if y*w.T.dot(X) < 1:
            grad_w = -y * X
        else:
            grad_w = np.zeros(w.shape)
        return grad_w

class LogLoss(object):
    def value(self, X, y, w):
        return np.log(1 + np.exp(-y*w.T.dot(X)))

    def grad(self, X, y, w):
        return -y*X / (1 + np.exp(y*w.T.dot(X)))


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    # Extend the features with the second power of each one
    X = np.hstack((X, np.log(np.absolute(X) + 1), X**2, np.absolute(X), np.sqrt(np.absolute(X))))
    # assert X.shape[1] == 800

    X = np.hstack((X, X**2))

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_norm = (X-x_mean) / x_std

    if RBF:
        rbf_sampler = RBFSampler(GAMMA, RBF_SPACE, 23)
        x_norm = rbf_sampler.fit_transform(x_norm)
    return x_norm

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
    svm = SGDClassifier(n_iter=ITERS, alpha=LAMBDA, loss=LOSS, penalty=REGULARIZATION,
        average=AVERAGING)
    svm.fit(X, Y)

    logger.info('Finish mapper')
    yield 'w', svm.coef_


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    w = np.vstack(values)
    w = np.mean(w,axis=0)
    yield w
