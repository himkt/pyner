from chainer import optimizer_hooks
from chainer import optimizers
import logging


logger = logging.getLogger(__name__)


def create_optimizer(optimizer_config):
    """
    :param optimizer_config: dict, 学習のパラメータを含む辞書
    """

    optimizer_ = optimizer_config['name']
    optimizer_ = optimizer_.lower()

    if optimizer_ == 'sgd':
        optimizer = optimizers.SGD(lr=optimizer_config['learning_rate'])

    elif optimizer_ == 'momentumsgd':
        optimizer = optimizers.MomentumSGD(
            lr=optimizer_config['learning_rate'])

    elif optimizer_ == 'adadelta':
        optimizer = optimizers.AdaDelta()

    elif optimizer_ == 'adam':
        optimizer = optimizers.Adam(alpha=optimizer_config['alpha'],
                                    beta1=optimizer_config['beta1'],
                                    beta2=optimizer_config['beta2'])

    else:
        raise Exception

    return optimizer


def add_hooks(optimizer, params):
    """
    :param optimizer: chainer.Optimizer, chainerのオプティマイザ
    :param params: dict, 学習のパラメータを含む辞書
    """

    if params.get('weight_decay'):
        logger.debug('set weight decay')
        optimizer.add_hook(optimizer_hooks.WeightDecay(
            params['weight_decay']))

    if params.get('gradient_clipping'):
        logger.debug('clip gradient')
        optimizer.add_hook(optimizer_hooks.GradientClipping(
            params['gradient_clipping']))

    return optimizer
