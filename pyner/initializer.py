from chainer.initializers import Uniform
from chainer import initializer

import numpy


class LampleUniform(initializer.Initializer):

    """Initializes array with hacky uniform distribution.
    https://github.com/glample/tagger/blob/master/utils.py#L44
    """

    def __init__(self, dtype=None):
        super(LampleUniform, self).__init__(dtype)

    def __call__(self, array):
        drange = numpy.sqrt(6. / (numpy.sum(array.shape)))
        Uniform()(array)  # destructive operation
        array *= drange
