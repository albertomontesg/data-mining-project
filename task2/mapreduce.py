import logging

import numpy as np

logger = logging.getLogger(__name__)

ITERS = 1
LAMBDA = .0001
LOSS = 'hinge'
REGULARIZATION = 'l2'
AVERAGING = False

RBF = True
GAMMA = 100
RBF_SPACE = 5000

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


class SGDClassifier(object):
    loss_functions = {
        'hinge': HingeLoss(),
        'log': LogLoss()
    }

    def __init__(self, n_iterations, lambda_, loss='log', penalty='l2', averaging=False, batch=1):
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
        self.averaging = averaging

    def _apply_regularization(self, w):
        if self.penalty == 'none':
            return 1.
        elif self.penalty == 'l1':
            return min(1.,
                       1. / (np.sqrt(self.lambda_) * np.linalg.norm(w, 1)))
        elif self.penalty == 'l2':
            return min(1.,
                       1. / (np.sqrt(self.lambda_) * np.linalg.norm(w, 2)))

    def fit(self, X, y):

        nb_samples = X.shape[0]
        nb_features = X.shape[1]
        assert nb_samples == y.shape[0]

        # Initialize weight vector to zero
        w_ = np.zeros((nb_features,), dtype='float')
        if self.averaging:
            w_avg = np.zeros((self.n_iterations*nb_samples, nb_features))
            a = 0

        for _ in range(self.n_iterations):
            # Shuffle data
            idx = np.random.permutation(nb_samples)
            X_ = X[idx,:]
            y_ = y[idx]

            for t in range(nb_samples):
                nhu = 1. / np.sqrt(t+1)
                w_ -= nhu * self.loss.grad(X_[t,:], y_[t], w_)
                # Scalar factor due to penalty
                w_ *= self._apply_regularization(w_)

                if self.averaging:
                    w_avg[a,:] = w_
                    a += 1

            # print('Done iteration {}'.format(i+1))
        if self.averaging:
            w_ = w_avg.mean(axis=0)


        self.w_ = w_

class RBFSampler(object):

    def __init__(self, gamma, n_components, seed):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

    def transform(self, X):
        n_features = X.shape[1]

        np.random.seed(self.seed)

        random_weights = (np.sqrt(2 * self.gamma) * np.random.normal(size=(n_features, self.n_components)))
        random_offset = np.random.uniform(0, 2 * np.pi, size=self.n_components)

        projection = X.dot(random_weights)
        projection += random_offset
        projection = np.cos(projection)
        projection *= np.sqrt(2) / np.sqrt(self.n_components)
        return projection


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    # Extend the features with the second power of each one
    if not RBF:
        X = np.hstack((X, np.log(np.absolute(X) + 1), X**2, np.absolute(X), np.sqrt(np.absolute(X))))
    else:
        rbf_sampler = RBFSampler(GAMMA, RBF_SPACE, 23)
        X = rbf_sampler.transform(X)

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_norm = (X-x_mean) / x_std

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
    svm = SGDClassifier(n_iterations=ITERS, lambda_=LAMBDA, loss=LOSS, penalty=REGULARIZATION,
        averaging=AVERAGING)
    svm.fit(X, Y)

    logger.info('Finish mapper')
    yield 'w', svm.w_


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    w = np.vstack(values)
    w = np.mean(w,axis=0)
    yield w
