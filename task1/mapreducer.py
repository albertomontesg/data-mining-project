import itertools

import numpy as np

# Definition
N = 8192 # Number of shingles

# Hash functions
b = 15
r = 15
print('\nb: {}\tr: {}'.format(b,r))
print('number of hash functions: {}'.format(r*b))
print('estimated threshold: {:.4f}'.format((1./float(b))**(1./float(r))))

# Hash parameters initializers
A = np.random.randint(0, high=8191, size=(r*b,))
B = np.random.randint(0, high=8191, size=(r*b,))
C = 131071      # Large Prime Number

# Hashing values for the use of the hash functions
def h(i, x):
    """ i is the hashing function that goes from 0 to r*b-1 and x is the position to hash """
    return ((A[i] * x + B[i]) % C) % N

def mapper(key, value):
    # key: None
    # value: one line of input file
    # Split the input line
    line = value.strip().split(' ')
    video_id = line[0]
    shingles = np.array(line[1:], dtype=np.int64)

    # Hashing
    M = np.full((r*b,), np.inf, dtype=np.float)
    for i in range(r*b):
        for row in shingles:
            M[i] = min(h(i, row), M[i])
    M = M.astype(np.int)

    for band in range(b):
        signature = M[band*r:(band+1)*r]
        k = '{:04d}'*r
        k = k.format(*signature)
        yield k, int(video_id[6:])


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    values = list(set(values))
    values.sort()

    if len(values) > 1:
        for k, v in itertools.combinations(values, 2):
            yield k,v

    if False:
        yield "key", "value"  # this is how you yield a key, value pair
