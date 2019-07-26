from pyner.vocab import load_sentences
from pyner.util import update_instances
from pathlib import Path

import chainer.dataset as D
import logging
import chainer
import numpy


logger = logging.getLogger(__name__)


def converter(batch, device=-1):
    '''
    convert the sequence of words into
    because it uses linear-chain CRF,
    output batch sorted by the size of them
    '''

    if device >= 0:
        xp = chainer.cuda.cupy

    else:
        xp = numpy

    x_array, t_array = list(zip(*batch))
    x_array = numpy.asarray([xp.asarray(x, dtype=xp.int32) for x in x_array])
    t_array = xp.asarray(t_array, dtype=xp.int32)
    return x_array, t_array


class DatasetTransformer:
    def __init__(self, vocab):
        self.char2idx = vocab.dictionaries['char2idx']
        self.label2idx = vocab.dictionaries['label2idx']
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def transform(self, word, label):
        word, = word  # input word is 1 sized list
        charids = [self.char2idx.get(c, self.char2idx['<UNK>'])
                   for c in word]

        label, = label  # input label is 1 sized list
        labelid = self.label2idx[label]
        return charids, labelid

    def itransform(self, charid_inputs, labelids):
        return [self._itransform(cids, lid) for cids, lid in
                zip(charid_inputs, labelids)]

    def _itransform(self, charids, labelid):
        chars = []
        for cid in charids.tolist():
            chars.append(self.idx2char.get(cid))
        word = ''.join(chars)
        label = self.idx2label[int(labelid)]
        return word, label


class WordClassDataset(D.DatasetMixin):
    def __init__(self, params, attr, transform):
        data_path = Path(params['data_dir'])
        words = load_sentences(data_path / f'{attr}.words.txt')
        labels = load_sentences(data_path / f'{attr}.labels.txt')
        datas = [words, labels]

        words, labels = update_instances(attr, datas, params)
        logger.debug(f'Created {attr} dataset ({len(words)} words are used)')

        self.words = words
        self.labels = labels

        self.transform = transform
        self.num_dataset = len(self.words)

    def __len__(self):
        return self.num_dataset

    def get_example(self, i):
        word = self.words[i]
        label = self.labels[i]
        return self.transform(word, label)
