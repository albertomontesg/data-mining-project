import itertools

import numpy as np

np.random.seed(23)

# Definition
N = 8192 # Number of shingles
m = 100000 # Number of buckets

# Hash functions for Signature Matrix
b = 12
r = 10
print('\nb: {}\tr: {}'.format(b,r))
print('number of hash functions: {}'.format(r*b))
print('estimated threshold: {:.4f}'.format((1./float(b))**(1./float(r))))

# Hash parameters initializers
A = np.random.randint(0, high=8191, size=(r*b,))
B = np.random.randint(0, high=8191, size=(r*b,))
A_s = np.random.randint(0, high=524287, size=(r,))
B_s = np.random.randint(0, high=524287, size=(r,))
C = 131071      # Large Prime Number
C_s = 524287

# Hashing values for the use of the hash functions
def h(i, x):
    """ i is the hashing function that goes from 0 to r*b-1 and x is the position to hash """
    return ((A[i] * x + B[i]) % C) % N

# Hashing column of the signature matrix
def h_s(x):
    """ x is the column vector of length r that is required to hash """
    return np.sum((A_s * x + B_s) % C_s) % m

def shingle_to_bitarray(shingle):
    bitarray = np.full((N,), False, dtype='bool')
    for s in shingle:
        bitarray[s] = True
    return bitarray

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
    # line = value.strip().split(' ')
    # video_id = line[0]
    # shingles = np.array(line[1:], dtype=np.int64)

    # Hashing
    M = np.full((r*b,), np.inf, dtype=np.float)
    for i in range(r*b):
        for row in shingles:
            M[i] = min(h(i, row), M[i])
    M = M.astype(np.int)

    for band in range(b):
        sig = M[band*r:(band+1)*r]
        k = '{:03d}_'.format(band+1)
        sig_hashed = h_s(sig)
        k += '{:010d}'.format(sig_hashed)
        # yield k, int(video_id[6:])
        yield k, value

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    values.sort()

    if len(values) > 1:
        for k, v in itertools.combinations(values, 2):
            k_id, k_shingle = parse_video_instance(k)
            v_id, v_shingle = parse_video_instance(v)

            # k_id = int(k[6:15])
            # v_id = int(v[6:15])
            # k_lines = k.strip().split(' ')
            # k_shingle = np.array(sorted(k_lines[1:]), dtype=np.int64)
            # v_lines = v.strip().split(' ')
            # v_shingle = np.array(sorted(v_lines[1:]), dtype=np.int64)

            # k_bits = shingle_to_bitarray(k_shingle)
            # v_bits = shingle_to_bitarray(v_shingle)
            # yield k_id, v_id
            if jacard_similarity(k_shingle, v_shingle) > .85:
                yield k_id, v_id
            # if jacard_similarity(k_bits, v_bits) > .85:
            #     yield k_id, v_id
