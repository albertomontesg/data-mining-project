import itertools

import numpy as np

#########################################################################
b = 3
r = 2
print('\nb: {}\tr: {}'.format(b,r))
print('number of hash functions: {}'.format(r*b))
print('estimated threshold: {:.4f}'.format((1./float(b))**(1./float(r))))
print('P(hit) = {:.4f}'.format(1-(1-.85**r)**b))
#########################################################################

class LSHashing(object):

    def __init__(self, r=20, b=50, nb_shingles=8192):
        self.r = r
        self.b = b
        self.nb_hash_functions = r*b
        self.N = nb_shingles
        self.N_b = 100000       # Number of buckets per band

        # Parameters for hashing shingles
        self.C_s = 131071
        self.A_s = np.random.randint(1, high=self.C_s, size=(r*b,))
        self.B_s = np.random.randint(0, high=self.C_s, size=(r*b,))

        # Parameters for hashing bands
        self.C_b = 524287
        self.A_b = np.random.randint(0, high=self.C_b, size=(r,))
        self.B_b = np.random.randint(0, high=self.C_b, size=(r,))

    def _h_s(self, i, x):
        """ i is the hashing function that goes from 0 to r*b-1 and x is the position to hash """
        return ((self.A_s[i] * x + self.B_s[i]) % self.C_s) % self.N

    def _h_b(self, x):
        """ x is the column vector of length r that is required to hash """
        return ((self.A_b * x + self.B_b) % self.C_b).sum() % self.N_b

    def hash(self, shingles):
        """ Return signature vector for the given shingle """
        M = np.full((self.nb_hash_functions,), np.inf, dtype=np.float)
        for i in range(self.nb_hash_functions):
            for row in shingles:
                M[i] = min(self._h_s(i, row), M[i])
        return M.astype(np.int)

    def hash_band(self, M):
        """ Return the hash of the signature column for each of the specified bands """
        band_hash = []
        for band in range(b):
            sig = M[band*self.r:(band+1)*self.r]
            band_hash.append(self._h_b(sig))
        return band_hash


def jacard_similarity(u, v):
    s1 = set(u)
    s2 = set(v)
    intersection_len = len(s1 & s2)
    union_len = len(s1 | s2)
    if intersection_len == 0:
        return 0.
    return float(intersection_len) / float(union_len)

def parse_video_instance(line):
    """ Return for each line describing a video, return its video_id and shingles array """
    video_id = int(line[6:15])
    shingles = np.array(line.strip().split(' ')[1:], dtype=np.int64)
    return video_id, shingles

def mapper(key, value):
    # key: None
    # value: one line of input file
    # Split the input line
    _, shingles = parse_video_instance(value)

    # Necessary to generate the same random numbers for hashing functions
    np.random.seed(seed=23)

    # Defining hashing object
    lsh = LSHashing(r, b)

    # Hashing shingles to signature matrix and then hashing the bands
    M = lsh.hash(shingles)
    B = lsh.hash_band(M)

    # For each hashed band return it as key with the value of the video
    for i in range(b):
        k = '{:03d}_{:010d}'.format(i, B[i])
        yield k, value


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    # Sort the values to then generate pairs with the key < value
    values.sort()

    if len(values) > 1:
        # In the case that more than one value has been passed to the reducer, it is
        # generated all the combinations of candidates pairs
        for k, v in itertools.combinations(values, 2):
            k_id, k_shingle = parse_video_instance(k)
            v_id, v_shingle = parse_video_instance(v)

            # For each candidate pair, it is computed the Jacard Similarity to return
            # only the pairs that are more similar than 85% and so, reduce the FP to
            # zero.
            if jacard_similarity(k_shingle, v_shingle) > .85:
                yield k_id, v_id
