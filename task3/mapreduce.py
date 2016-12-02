import logging

import numpy as np

logger = logging.getLogger(__name__)

DEBUG = 2
N_CLUSTERS = 200
N_INIT = 3

np.random.seed(23)

def euclidean_distance(X, Y):
    if len(Y.shape) == 1:
        return np.sum((X-Y)**2, axis=1)

    assert X.shape[1] == Y.shape[1], 'Last dimension do not match'

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        result[i,:] = euclidean_distance(Y, X[i,:])
    return result

class KMeans(object):

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=.0001):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def _kmeans_pp_init(self, X):
        # Centers container
        centers = np.zeros((self.n_clusters, X.shape[-1]))
        # Choose the first center
        centers[0,:] = X[np.random.randint(X.shape[0]),:]

        n_local_trials = 2*int(np.log(self.n_clusters))

        closest_dist_sq = np.sum((X-centers[0])**2, axis=1)
        current_pot = closest_dist_sq.sum()

        for c in range(self.n_clusters - 1):
            rand_vals = np.random.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

            distance_to_candidates = euclidean_distance(X[candidate_ids], X)
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                new_dist_sq = np.minimum(closest_dist_sq,
                                         distance_to_candidates[trial])
                new_pot = new_dist_sq.sum()

                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[c] = X[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        if DEBUG >= 2:
            logger.info('Initialization kmeans++ finished')
        return centers

    def fit(self, X):

        best_centers, best_inertia, best_labels = None, None, None

        n_samples = X.shape[0]


        for i in range(self.n_init):

            # Initialize the centers choosing randomly n_clusters points
            # centers = X[np.random.permutation(X.shape[0])[:self.n_clusters],:]
            centers = self._kmeans_pp_init(X)

            it = 0
            prev_L = 0
            while it < self.max_iter:

                L = 0
                # Assign to each point the index of the closest center
                labels = np.zeros(n_samples, dtype='int')
                for j in range(n_samples):
                    d_2 = np.sum((centers-X[j,:])**2, axis=1)
                    labels[j] = np.argmin(d_2)
                    L += np.min(d_2)
                L /= n_samples

                # Update
                for l in range(self.n_clusters):
                    P = X[labels==l,:]
                    centers[l] = np.mean(P, axis=0)

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

class KMeansCoresets(object):

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=.0001):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def _kmeans_pp_init(self, X):
        """ k-means++ algorithm for initialize centers """
        # Centers container
        centers = np.zeros((self.n_clusters, X.shape[-1]))
        # Choose the first center
        centers[0,:] = X[np.random.randint(X.shape[0]),:]

        n_local_trials = 2*int(np.log(self.n_clusters))

        closest_dist_sq = np.sum((X-centers[0])**2, axis=1)
        current_pot = closest_dist_sq.sum()

        for c in range(self.n_clusters - 1):
            rand_vals = np.random.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

            distance_to_candidates = euclidean_distance(X[candidate_ids], X)
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                new_dist_sq = np.minimum(closest_dist_sq,
                                         distance_to_candidates[trial])
                new_pot = new_dist_sq.sum()

                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[c] = X[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        if DEBUG >= 2:
            logger.info('Initialization kmeans++ finished')
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
            centers = self._kmeans_pp_init(X)

            it = 0
            prev_L = 0
            while it < self.max_iter:

                L = 0
                # Assign to each point the index of the closest center
                labels = np.zeros(n_samples, dtype='int')
                for j in range(n_samples):
                    d_2 = np.sum((centers-X[j,:])**2, axis=1)
                    labels[j] = np.argmin(d_2)
                    L += w[i] * np.min(d_2)
                L /= n_samples

                # Update
                for l in range(self.n_clusters):
                    P = X[labels==l,:]
                    pw = w[labels==l]
                    centers[l] = np.mean(pw * P, axis=0)

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

    print(value.shape)

    # To compute the coreset first uniformly sample over all the points.
    # Then compute the weight of each sample as the number of samples closer to each one
    X = value
    idx = np.random.permutation(X.shape[0])
    C = X[idx[:1000]]

    labels = np.zeros((C.shape[0],))
    for i in range(X.shape[0]):
        d = euclidean_distance(C, X[i])
        labels[np.argmin(d)] += 1

    if DEBUG >= 1:
        logger.info('Finish mapper')
    yield 'w', C


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    print(values.shape)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT)
    kmeans.fit(values)


    yield kmeans.cluster_centers_
