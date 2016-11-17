import numpy as np

LAMBDA = .01

np.random.seed(23)

def _dot(*args):
    """ Perform the dot product to the given arguments in order.
    It avoids writing long chains of dot products """
    assert len(args) >= 2, 'Not enough arguments'
    out = np.dot(args[0], args[1])
    for a in args[2:]:
        out = np.dot(out, a)
    return out


def _hinge_loss_and_gradient(X, y, w):
    loss = np.fmax(np.zeros(X.shape[0],), 1-y*_dot(w, X.T))
    grad_W = y.reshape(-1, 1) * X
    grad_W[loss==0] = 0
    return loss, grad_W


class Trainer(object):

    def __init__(self, n_iterations, batch_size, lambda_):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_ = lambda_

    def _batch_iter(self, nb_samples):
        initial_pointer = 0
        for i in range(self.n_iterations):
            if initial_pointer == 0:
                # Shuffle X at the begining
                idx = np.random.permutation(nb_samples)

            end_pointer = min(initial_pointer+self.batch_size, nb_samples)

            yield i, idx[initial_pointer:end_pointer]
            initial_pointer = end_pointer % nb_samples

    def fit(self, X, y):
        assert len(X.shape) == 2, 'X shape invalid'
        self.w_ = np.empty((X.shape[1],))

        self._init_training(self.w_)

        nb_samples = X.shape[0]

        for t, idx in self._batch_iter(nb_samples):
            X_ = X[idx]
            y_ = y[idx]
            self._train_iter(t, self.w_, X_, y_)
            self.w_ *= min(1.,
                     1. / (np.sqrt(self.lambda_) * np.linalg.norm(self.w_, 2)))


class PEGASOS(Trainer):

    def __init__(self, lambda_, n_iterations, batch_size=32):
        super(PEGASOS, self).__init__(n_iterations, batch_size, lambda_)

    def _init_training(self, w):
        w[:] = 0.

    def _train_iter(self, t, w, X, y):
        nhu = 1. / ((t+1) * self.lambda_)

        loss, grad = _hinge_loss_and_gradient(X, y, w)
        grad_W = self.lambda_ * w - nhu / X.shape[0] * np.sum(grad[loss > 0], axis=0)
        w -= nhu * grad_W

class SGD(Trainer):

    def __init__(self, n_iterations, batch_size=1, lambda_=LAMBDA):
        super(SGD, self).__init__(n_iterations, batch_size, lambda_)

    def _init_training(self, w):
        w[:] = 0.

    def _train_iter(self, t, w, X, y):
        nhu = 1. / np.sqrt(t+1)

        _, grad_W = _hinge_loss_and_gradient(X, y, w)

        w += nhu * np.mean(grad_W, axis=0)

class SGDMomentum(Trainer):

    def __init__(self, n_iterations, batch_size=1, lambda_=LAMBDA, momentum=.9):
        self.momentum = momentum
        super(SGDMomentum, self).__init__(n_iterations, batch_size, lambda_)

    def _init_training(self, w):
        w[:] = 0.
        self.g = np.zeros(w.shape)

    def _train_iter(self, t, w, X, y):
        nhu = 1. / np.sqrt(t+1)

        _, grad_W = _hinge_loss_and_gradient(X, y, w)

        self.g = self.g*self.momentum + (1-self.momentum)*np.mean(grad_W, axis=0)

        w += nhu * self.g

class Adam(Trainer):

    def __init__(self, n_iterations, batch_size=32, lambda_=LAMBDA, learning_rate=0.01, beta_1=.9, beta_2=.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        super(Adam, self).__init__(n_iterations, batch_size, lambda_)

    def _init_training(self, w):
        w[:] = 0.
        self.m = np.zeros(w.shape, dtype='float')
        self.v = np.zeros(w.shape, dtype='float')

    def _train_iter(self, t, w, X, y):
        _, grad_W = _hinge_loss_and_gradient(X, y, w)
        grad_W = grad_W.mean(axis=0)
        t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta_2**t) / (1 - self.beta_1**t)

        self.m = self.beta_1*self.m + (1-self.beta_1) * grad_W
        self.v = self.beta_2*self.v + (1-self.beta_2) * grad_W**2

        w += lr_t * self.m / (np.sqrt(self.v) + self.epsilon)

class HingeLoss(object):
    @classmethod
    def value(self, X, y, w):
        return np.fmax(np.zeros(X.shape[0],), 1-y*_dot(w, X.T))

    @classmethod
    def grad(self, X, y, w):
        if y*w.T.dot(X) < 1:
            grad_w = y * X
        else:
            grad_w = np.zeros(w.shape)
        return grad_w

class LogLoss(object):
    @classmethod
    def value(self, X, y, w):
        return np.log(1+np.exp(-y.dot(w.T).dot(X)))

    @classmethod
    def grad(self, X, y, w):
        return y.dot(X) / (1 + np.exp(y.dot(w).dot(X)))

class SGD(Trainer):

    def __init__(self, n_iterations, batch_size=1, lambda_=LAMBDA):
        super(SGD, self).__init__(n_iterations, batch_size, lambda_)

    def _init_training(self, w):
        w[:] = 0.

    def _train_iter(self, t, w, X, y):
        nhu = 1. / np.sqrt(t+1)

        _, grad_W = _hinge_loss_and_gradient(X, y, w)

        w += nhu * np.mean(grad_W, axis=0)


class SGDClassifier(object):
    loss_functions = {
        'hinge': HingeLoss,
        'log': LogLoss
    }

    def __init__(self, n_iterations, lambda_, loss='hinge', penalty='l2'):
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
        if self.penalty == 'none':
            return w
        elif self.penalty == 'l1':
            return w * min(1.,
                           1. / (np.sqrt(self.lambda_) * np.linalg.norm(w, 1)))
        elif self.penalty == 'l2':
            return w * min(1.,
                           1. / (np.sqrt(self.lambda_) * np.linalg.norm(w, 2)))

    def fit(self, X, y):

        nb_samples = X.shape[0]
        nb_features = X.shape[1]
        assert nb_samples == y.shape[0]

        # Initialize weight vector to zero
        w_ = np.zeros((nb_features,), dtype='float')

        for i in range(self.n_iterations):
            # Shuffle data
            idx = np.random.permutation(nb_samples)
            X = X[idx,:]

            for t in range(nb_samples):
                nhu = nhu = 1. / np.sqrt(t+1)
                w_ += nhu * self.loss.grad(X[t], y[t], w_)

                w_ = self._apply_regularization(w_)


            print('Done iteration {}'.format(i+1))

        self.w_ = w_

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    # Extend the features with the second power of each one
    X = np.hstack((X, np.log(np.absolute(X) + 1), X**2, np.absolute(X), np.sqrt(np.absolute(X))))
    # assert X.shape[1] == 800

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
    svm = SGDClassifier(n_iterations=1, lambda_=LAMBDA, loss='hinge', penalty='l2')
    svm.fit(X,Y)

    print('Finish one mapper')
    yield 'w', svm.w_


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    w = np.vstack(values)
    w = np.mean(w,axis=0)
    yield w
