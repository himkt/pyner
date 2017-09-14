import argparse
import collections

import nltk
import numpy

import chainer
from chainer import datasets
from chainer import training
from chainer.training import extensions

from model import CRFTagger
from model import BiLSTMTagger
from model import BiLSTMCRFTagger


def convert(batch, device):
    sentences = [
        chainer.dataset.to_device(device, sentence) for sentence, _ in batch]
    poses = [chainer.dataset.to_device(device, pos) for _, pos in batch]
    return {'xs': sentences, 'ys': poses}


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: POS-tagging')
    parser.add_argument('--batchsize', '-b', type=int, default=30,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    vocab = collections.defaultdict(lambda: len(vocab))
    pos_vocab = collections.defaultdict(lambda: len(pos_vocab))

    # Convert word sequences and pos sequences to integer sequences.
    nltk.download('brown')
    data = []
    for sentence in nltk.corpus.brown.tagged_sents():
        xs = numpy.array([vocab[lex] for lex, _ in sentence], 'i')
        ys = numpy.array([pos_vocab[pos] for _, pos in sentence], 'i')
        data.append((xs, ys))

    print('# of sentences: {}'.format(len(data)))
    print('# of words: {}'.format(len(vocab)))
    print('# of pos: {}'.format(len(pos_vocab)))

    # model = CRFTagger(len(vocab), len(pos_vocab))
    # model = BiLSTMTagger(n_dim=300, n_vocab=len(vocab),
    #                      n_pos=len(pos_vocab), n_hidden=200)
    model = BiLSTMCRFTagger(n_dim=300, n_vocab=len(vocab),
                            n_pos=len(pos_vocab), n_hidden=200)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    test_data, train_data = datasets.split_dataset_random(
        data, len(data) // 10, seed=0)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = extensions.Evaluator(
        test_iter, model, device=args.gpu, converter=convert)
    # Only validate in each 1000 iteration
    trainer.extend(evaluator, trigger=(1000, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')),
                   trigger=(100, 'iteration'))

    trainer.extend(
        extensions.MicroAverage(
            'main/correct', 'main/total', 'main/accuracy'))
    trainer.extend(
        extensions.MicroAverage(
            'validation/main/correct', 'validation/main/total',
            'validation/main/accuracy'))

    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
        trigger=(100, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
