import chainer.optimizers as Opt
from pathlib import Path

import argparse
import chainer
import random
import numpy
import operator
import logging
import json


logger = logging.getLogger(__name__)


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--gpu', type=int, default=-1, dest='device')
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    return args


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--gpu', type=int, dest='device', default=-1)
    parser.add_argument('--metric', default='validation/main/fscore')
    args = parser.parse_args()
    return args


def select_snapshot(args, model_dir):
    if args.epoch is None:
        epoch, max_value = argmax_metric(model_dir / 'log', args.metric)
        logger.debug(f'Epoch is {epoch:04d} ({args.metric}: {max_value:.2f})')  # NOQA
        metric_repr = args.metric.replace('/', '.')
        prediction_path = Path(args.model, f'{metric_repr}.epoch_{epoch:03d}.pred')  # NOQA

    else:
        epoch = args.epoch
        logger.debug(f'Epoch is {epoch:04d} (which is specified manually)')
        prediction_path = Path(args.model, f'epoch_{epoch:03d}.pred')

    snapshot_file = f'snapshot_epoch_{epoch:04d}'
    return snapshot_file, prediction_path


def update_instances(train_datas, params):
    train_size = params.get('train_size', 1.0)
    if train_size <= 0 or 1 <= train_size:
        assert Exception('train_size must be in (0, 1]')
    n_train = len(train_datas[0])
    instances = int(train_size * n_train)
    rate = 100 * train_size
    logger.debug(f'Use {instances} example for training ({rate:.2f}%)')
    return [t[:instances] for t in train_datas]


def argmax_metric(log_file, metric):
    op = prepare_op(metric)
    best_epoch = 0

    if op == operator.ge:
        best_value = -1_001_001_001

    elif op == operator.le:
        best_value = 1_001_001_001

    documents = json.load(open(log_file))
    for document in documents:
        value = document[metric]
        epoch = document['epoch']

        if op(value, best_value):
            best_epoch = epoch
            best_value = value

    return best_epoch, best_value


def prepare_op(metric):
    ge_metrics = ['accuracy', 'precision', 'recall', 'fscore']
    le_metrics = ['loss']

    for ge_metric in ge_metrics:
        if ge_metric in metric:
            return operator.ge

    for le_metric in le_metrics:
        if le_metric in metric:
            return operator.le

    raise NotImplemented


def set_seed(seed=31, device=-1):
    logger.debug(f'Seed value: {seed}')

    if chainer.cuda.available and device >= 0:
        logger.debug('Fix cupy random seed')
        chainer.cuda.cupy.random.seed(seed)

    logger.debug('Fix numpy random seed')
    numpy.random.seed(seed)
    logger.debug('Fix random seed')
    random.seed(seed)


def create_optimizer(params):
    """
    :param params: dict, 学習のパラメータを含む辞書
    """

    optimizer_ = params['optimizer']
    optimizer_ = optimizer_.lower()

    if optimizer_ == 'sgd':
        optimizer = Opt.SGD(lr=params['learning_rate'])

    elif optimizer_ == 'momentumsgd':
        optimizer = Opt.MomentumSGD(lr=params['learning_rate'])

    elif optimizer_ == 'adadelta':
        optimizer = Opt.AdaDelta()

    elif optimizer_ == 'adam':
        optimizer = Opt.Adam(alpha=params['learning_rate'],
                             beta1=0.9, beta2=0.9)

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
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(
            params['weight_decay']))

    if params.get('gradient_clipping'):
        logger.debug('clip gradient')
        optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(
            params['gradient_clipping']))

    return optimizer
