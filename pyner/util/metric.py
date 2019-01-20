from pathlib import Path

import operator
import logging
import json


logger = logging.getLogger(__name__)


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

    raise NotImplementedError
