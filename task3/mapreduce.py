import logging

import numpy as np

logger = logging.getLogger(__name__)

DEBUG = 2
N_CLUSTERS = 200
N_INIT = 20
N_CORESETS = 500
NOTES = """
For sampling the Coresets centers, used the D^2 sampling criteria.
The initialization of the kmeans algorithm is using uniformly sampling.
The best results have been obtained increasing the number of coresets return by each mapper at
least to 400 coresets per mapper.
"""

print('\n' + '#'*40 + '\n')
print('Number of clusters:\t\t\t{}'.format(N_CLUSTERS))
print('Number of initializations:\t{}'.format(N_INIT))
print('Number of coresets:\t\t\t{}'.format(N_CORESETS))
print('Notes: {}'.format(NOTES))
print('\nResult:')

np.random.seed(23)

def euclidean_distance(X, Y):
    """ Compute the euclidean distance pair-wise between column vectors of X and Y """
    if len(Y.shape) == 1:
        return np.sum((X-Y)**2, axis=1)

    assert X.shape[1] == Y.shape[1], 'Last dimension do not match'

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        result[i,:] = euclidean_distance(Y, X[i,:])
    return result

def init_centers_d2_sampling(X, n_clusters):
    """ D^2 sampling algorithm to find a proper cluster centers given a dataset of points """
    # Extract some values
    N, d = X.shape

    # Centers container
    centers = np.empty((n_clusters, d), dtype=X.dtype)
    # Squared distance of each point to its closest center
    dist = np.empty((N, n_clusters), dtype='float')

    indexes = np.arange(N)
    p = np.ones((N))
    p /= p.sum()

    for i in range(n_clusters):
        # Sample following the given probability distribution
        idx = np.random.choice(indexes, p=p)
        # And store the sampled point into the centers container
        centers[i] = X[idx]

        # Squared distance of each point to its closest center
        dist[:,i] = euclidean_distance(X, centers[i])
        min_dist = dist[:,:i+1].min(axis=1) # Distance to closest center

        # Compute the probability distribution normalizing the squared distance
        p = min_dist / min_dist.sum()

        if DEBUG > 2:
            logger.info('Sampled center {}'.format(i))

    if DEBUG > 1:
        logger.info('Finish D^2 sampling')

    return centers

def init_centers_random_unif(X, n_clusters):
    """ Initialize choosing randomly samples as centers """
    n_samples = X.shape[0]

    idx = np.random.permutation(n_samples)[:n_clusters]
    centers = X[idx,:]
    return centers

def kmeans_coresets(X, w, n_clusters=8, n_init=10, max_iter=300, tol=.0001):
    """ Fit the K-Means cluster algorithm with the coresets represented by the points `X` and
    weights `w` """

    assert X.shape[0] == w.shape[0], \
        "X and w must have the same number of samples. {} != {}".format(X.shape[0], w.shape[0])

    best_centers, best_inertia, best_labels = None, None, None

    n_samples = X.shape[0]

    for i in range(n_init):

        # Initialize the centers using the k-means++ algorithm
        centers = init_centers_random_unif(X, n_clusters)

        it = 0
        prev_L = 0
        while it < max_iter:

            L = 0
            # Assign to each point the index of the closest center
            labels = np.zeros(n_samples, dtype='int')
            for j in range(n_samples):
                d_2 = np.sum((centers-X[j,:])**2, axis=1)
                labels[j] = np.argmin(d_2)
                L += w[i,0] * d_2[labels[j]]
            L /= w.sum()

            # Update
            for l in range(n_clusters):
                if np.sum(labels==l) == 0:
                    logger.warning('No labels of {}'.format(l))
                    continue
                P = X[labels==l,:]
                pw = w[labels==l,:]
                centers[l] = np.sum(pw * P, axis=0) / pw.sum()

            # Check convergence
            if abs(prev_L - L) < tol:
                break
            prev_L = L

            if DEBUG >= 2:
                logger.info('Iteration {}\tInertia: {:.5f}'.format(it, L))
            it += 1

        if it == max_iter:
            logger.warning('Maximum iteration reached')
        elif DEBUG >= 1:
            logger.info('Finished initialization {} with {} iterations and intertia {:.5f}'.format(i, it, L))
        # Compute intertia and update the best parameters
        inertia = L
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return best_centers


def mapper(key, value):
    # key: None
    # value: one line of input file
    assert key is None, 'key is not None'

    # To compute the coreset first uniformly sample over all the points.
    # Then compute the weight of each sample as the number of samples closer to each one
    X = value
    nb_samples = X.shape[0]

    # Sample points for coresets
    c = init_centers_d2_sampling(X, N_CORESETS)

    w = np.zeros((N_CORESETS, 1))
    for i in range(nb_samples):
        d = euclidean_distance(c, X[i])
        w[np.argmin(d), 0] += 1

    C = np.hstack([w, c])

    if DEBUG >= 1:
        logger.info('Finished mapper')
    yield 'w', C


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    w, X = values[:,0].reshape(-1, 1), values[:,1:]
    cluster_centers = kmeans_coresets(X, w, N_CLUSTERS, N_INIT)

    yield cluster_centers
