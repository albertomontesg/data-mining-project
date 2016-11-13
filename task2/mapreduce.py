import numpy as np

SEED = 23
LAMBDA = .01
GAMMA = 250
RBF_SPACE = 1000 # N components of the projection of x to rbf space

class OnlineSVM(object):

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, X, y):
        assert len(X.shape) == 2, 'X shape invalid'
        self.w_ = np.zeros((X.shape[1],))

        nb_samples = X.shape[0]

        for t in range(nb_samples):
            nhu = 1. / np.sqrt(t+1)
            if np.dot(y[t], np.dot(self.w_, X[t])) < 1:
                self.w_ += nhu * np.dot(y[t], X[t])
                self.w_ *= min(1.,
                              1. / (np.sqrt(self.lambda_) *
                                    np.linalg.norm(self.w_, 2)))

class OnlineLogisticRegression(object):

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, X, y):
        assert len(X.shape) == 2, 'X shape invalid'
        self.w_ = np.zeros((X.shape[1],))

        nb_samples = X.shape[0]

        for t in range(nb_samples):
            nhu = 1. / np.sqrt(t+1)

            self.w_ += nhu * np.dot(y[t], X[t]) / (1 + np.exp(np.dot(y[t], np.dot(self.w_, X[t]))))
            self.w_ *= min(1.,
                          1. / (np.sqrt(self.lambda_) * np.linalg.norm(self.w_, 1)))


class RBFSampler(object):

    def __init__(self, gamma, n_components, random_state=SEED):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def transform(self, X):
        nb_features = X.shape[1]
        np.random.seed(self.random_state)
        self.random_weights_ = (np.sqrt(2 * self.gamma) * np.random.normal(
            size=(nb_features, self.n_components)))
        self.random_offset_ = np.random.uniform(0, 2*np.pi, size=self.n_components)

        projection = np.dot(X, self.random_weights_)
        projection += self.random_offset_
        projection = np.cos(projection)
        projection += np.sqrt(2.) / np.sqrt(self.n_components)
        return projection


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    x_mean = X.mean(axis=1, keepdims=True)
    x_std = X.std(axis=1, keepdims=True)
    x_norm = (X-x_mean) / x_std
    return x_norm
    # rbfsampler = RBFSampler(GAMMA, RBF_SPACE, SEED)
    # x_proj = rbfsampler.transform(x_norm)
    #
    # return x_proj

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
    svm = OnlineSVM(LAMBDA)
    svm.fit(X,Y)

    print('Finish one mapper')
    yield 'w', svm.w_  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    w = np.vstack(values)
    w = np.mean(w,axis=0)
    yield w
