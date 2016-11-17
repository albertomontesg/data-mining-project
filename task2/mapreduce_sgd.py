import numpy as np
from sklearn.linear_model import SGDClassifier

LAMBDA = .01

np.random.seed(23)

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    # Extend the features with the second power of each one
    X = np.hstack((X, np.log(np.absolute(X) + 1), X**2, np.absolute(X), np.sqrt(np.absolute(X))))
    # assert X.shape[1] == 1600


    # x_mean = X.mean(keepdims=True)
    # x_std = X.std(keepdims=True)
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

    svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, fit_intercept=False,
        n_iter=10, shuffle=True, random_state=23, verbose=False)
    svm.fit(X,Y)

    print('Finish one mapper')
    yield 'w', svm.coef_


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    w = np.vstack(values)
    w = np.mean(w,axis=0)
    yield w
