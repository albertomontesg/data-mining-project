import logging

import numpy as np

logger = logging.getLogger(__name__)


np.random.seed(23)


def mapper(key, value):
    # key: None
    # value: one line of input file
    assert key is None, 'key is not None'

    print(value.shape)

    logger.info('Finish mapper')
    yield 'w', None


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

    yield np.random.randn(200, 250)
