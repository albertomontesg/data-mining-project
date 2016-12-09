import logging

import numpy as np

logger = logging.getLogger(__name__)

DEBUG = 2
N_CLUSTERS = 200
N_INIT = 10
N_CORESETS = 500
NOTES = """
For sampling the Coresets centers, used the D^2 sampling criteria.

(At the previous experiments): The initialization of the kmeans operation at the reducer was using
uniformly sampling between all the points. Now it is done again as the initialization of the
centers do not take into account the weights, so better to randomly sample the centers using a
sample weight proportional to coresets weight.

Increassing the number of coresets bigger than 400 it is obtained better results than the hard
baseline
"""

print('\n' + '#'*40 + '\n')
print('Number of clusters:\t\t\t{}'.format(N_CLUSTERS))
print('Number of initializations:\t{}'.format(N_INIT))
print('Number of coresets:\t\t\t{}'.format(N_CORESETS))
print('Notes: {}'.format(NOTES))
print('\nResult:')

np.random.seed(23)

def euclidean_distance(X, Y):
    if len(Y.shape) == 1:
        return np.sum((X-Y)**2, axis=1)

    assert X.shape[1] == Y.shape[1], 'Last dimension do not match'

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        result[i,:] = euclidean_distance(Y, X[i,:])
    return result

def d_2_sampling(X, n_clusters):
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
        logger.info('Finish kmeans++ initialization')

    return centers

def importance_sampling(X, n_centers):
    """ Implementation of the importance sampling """
    # Extract some values
    N, d = X.shape

    # Centers container
    centers = np.empty((n_centers, d), dtype=X.dtype)
    # Squared distance of each point to its closest center
    dist = np.empty((N, n_centers), dtype='float')

    alpha = 1 + np.log2(n_centers)

    # Initial sampling distribution (uniformly)
    indexes = np.arange(N)
    q = np.ones(N) * 1 / N

    for i in range(n_centers):
        # Sample following the given probability distribution
        idx = np.random.choice(indexes, p=q)
        # And store the sampled point into the centers container
        centers[i] = X[idx]

        # Squared distance of each point to its closest center
        dist[:,i] = euclidean_distance(X, centers[i])
        min_dist = dist[:,:i+1].min(axis=1) # Distance to closest center

        c_phi = min_dist.mean()
        q = alpha * min_dist / c_phi

        min_i = np.argmin(dist[:,:i+1], axis=1)

        for j in range(N):
            idx_B = min_i == min_i[j]
            q[j] += 2 * alpha * min_dist[idx_B].sum() / (idx_B.sum() * c_phi)

            q[j] += 4 * N / idx_B.sum()

        # Normalize sample weights
        q /= q.sum()

        if DEBUG > 2:
            logger.info('Sampled center {}'.format(i))

    if DEBUG > 1:
        logger.info('Finish importance sampling')

    return centers

class KMeansCoresets(object):

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=.0001):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def init_centers(self, X):
        """ Initialize choosing randomly samples as centers """
        n_samples = X.shape[0]

        idx = np.random.permutation(n_samples)[:self.n_clusters]
        centers = X[idx,:]
        return centers

    def fit(self, X, w):
        """ Fit the K-Means cluster algorithm with the coresets represented by the points `X` and
        weights `w` """

        assert X.shape[0] == w.shape[0], \
            "X and w must have the same number of samples. {} != {}".format(X.shape[0], w.shape[0])

        best_centers, best_inertia, best_labels = None, None, None

        n_samples = X.shape[0]

        for i in range(self.n_init):

            # Initialize the centers using the k-means++ algorithm
            centers = self.init_centers(X)

            it = 0
            prev_L = 0
            while it < self.max_iter:

                L = 0
                # Assign to each point the index of the closest center
                labels = np.zeros(n_samples, dtype='int')
                for j in range(n_samples):
                    d_2 = np.sum((centers-X[j,:])**2, axis=1)
                    labels[j] = np.argmin(d_2)
                    L += w[i,0] * d_2[labels[j]]
                L /= w.sum()

                # Update
                for l in range(self.n_clusters):
                    if np.sum(labels==l) == 0:
                        logger.warning('No labels of {}'.format(l))
                        continue
                    P = X[labels==l,:]
                    pw = w[labels==l,:]
                    centers[l] = np.sum(pw * P, axis=0) / pw.sum()

                # Check convergence
                if abs(prev_L - L) < self.tol:
                    break
                prev_L = L

                if DEBUG >= 2:
                    logger.info('Iteration {}\tInertia: {:.5f}'.format(it, L))
                it += 1

            if it == self.max_iter:
                logger.warning('Maximum iteration reached')
            elif DEBUG >= 1:
                logger.info('Finished initialization {} with {} iterations and intertia {:.5f}'.format(i, it, L))
            # Compute intertia and update the best parameters
            inertia = L
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = best_centers


def mapper(key, value):
    # key: None
    # value: one line of input file
    assert key is None, 'key is not None'

    # To compute the coreset first uniformly sample over all the points.
    # Then compute the weight of each sample as the number of samples closer to each one
    X = value
    nb_samples = X.shape[0]

    # Sample points for coresets
    c = d_2_sampling(X, N_CORESETS)

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

    w, x = values[:,0].reshape(-1, 1), values[:,1:]
    kmeans = KMeansCoresets(n_clusters=N_CLUSTERS, n_init=N_INIT)
    kmeans.fit(x, w)

    yield kmeans.cluster_centers_
