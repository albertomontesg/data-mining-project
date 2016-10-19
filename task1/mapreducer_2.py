import itertools

import numpy as np

# Necessary to generate the same random numbers for hashing functions
np.random.seed(seed=23)

# Definition
N = 8192 # Number of shingles

# Hash functions for Signature Matrix
b = 5
r = 20
print('\nb: {}\tr: {}'.format(b,r))
print('number of hash functions: {}'.format(r*b))
print('estimated threshold: {:.4f}'.format((1./float(b))**(1./float(r))))


class LSHashing(object):

    def __init__(self, r=20, b=50, nb_shingles=8192):
        self.r = r
        self.b = b
        self.nb_hash_functions = r*b
        self.N = nb_shingles
        self.N_b = 200000

        # Parameters for hashing shingles
        self.C_s = 131071
        self.A_s = np.random.randint(1, high=self.C_s, size=(r*b,))
        self.B_s = np.random.randint(0, high=self.C_s, size=(r*b,))

        # Parameters for hashing bands
        self.C_b = 524287
        self.A_b = np.random.randint(0, high=self.C_s, size=(r,))
        self.B_b = np.random.randint(0, high=self.C_s, size=(r,))

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


def mapper(key, value):
    # key: None
    # value: one line of input file

    # Split the input line
    line = value.strip().split(' ')
    video_id = line[0]
    shingles = np.array(sorted(line[1:]), dtype=np.int64)

    lsh = LSHashing(r, b, N)
    # Hash the shingles and obtain the signature column for this instance
    M = lsh.hash(shingles)
    # Hash each band into buckets
    B = lsh.hash_band(M)

    for i in range(b):
        b_h = B[i]
        # Create a key with the band index and the bucket value to return
        k = '{:03d}_{:06d}'.format(i, b_h)
        yield k, value


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    values = list(set(values))
    values.sort()

    if len(values) > 1:
        for k, v in itertools.combinations(values, 2):
            k_id = int(k[6:15])
            v_id = int(v[6:15])
            yield k_id, v_id
