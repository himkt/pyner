import logging
import random

import chainer
import numpy

logger = logging.getLogger(__name__)


def set_seed(seed=31, device=-1):
    logger.debug(f"Seed value: {seed}")

    if chainer.cuda.available and device >= 0:
        logger.debug("Fix cupy random seed")
        chainer.cuda.cupy.random.seed(seed)

    logger.debug("Fix numpy random seed")
    numpy.random.seed(seed)
    logger.debug("Fix random seed")
    random.seed(seed)
