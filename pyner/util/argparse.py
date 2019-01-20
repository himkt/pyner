import argparse
import logging


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
