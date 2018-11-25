from pathlib import Path

import logging
import gensim
import numpy
import re


logger = logging.getLogger(__name__)
SPECIAL_SYMBOLS = ['<UNK>', '<PAD>']


def load_sentences(file):
    sentences = []
    with open(file, encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n')
            sentences.append(line.split(' '))
    return sentences


def _lower(ws):
    ws = [w if w in SPECIAL_SYMBOLS else w.lower() for w in ws]
    return ws


def _replace_zero(ws):
    ws = [re.sub(r'\d', '0', w) for w in ws]
    return ws


def load_pretrained_word_vector(word2idx, word_vector):
    gensim_model = gensim.models.KeyedVectors.load(word_vector)
    word_dim = gensim_model.wv.vector_size

    n_word_vocab = len(word2idx)
    scale = numpy.sqrt(3.0 / word_dim)
    shape = [n_word_vocab, word_dim]

    # if lowercased word is in pre-trained embeddings,
    # increment match2
    match1, match2 = 0, 0

    word_vectors = numpy.random.uniform(-scale, scale, shape)
    for word, idx in word2idx.items():
        if word in gensim_model:
            word_vector = gensim_model.wv.word_vec(word)
            word_vectors[idx, :] = word_vector
            match1 += 1

        elif word.lower() in gensim_model:
            word_vector = gensim_model.wv.word_vec(word.lower())
            word_vectors[idx, :] = word_vector
            match2 += 1

    match = match1 + match2
    matching_rate = 100 * (match/n_word_vocab)
    logger.info(f'Found {matching_rate:.2f}% words in pre-trained vocab')
    logger.info(f'- n_word_vocab: {n_word_vocab}')
    logger.info(f'- match1: {match1}, match2: {match2}')
    return word_vectors


class Vocabulary:
    def __init__(self, params):
        self.__name__ = 'Vocabulary'

        # use self.replace_zero in Dataset
        if 'replace_zero' not in params:
            self.replace_zero = False
        else:
            self.replace_zero = params['replace_zero']

        if 'lower' not in params:
            self.lower = False
        else:
            self.lower = params['lower']

        self.params = params
        self.data_path = Path(params['data_dir'])
        self.dictionaries = {}
        self.vocab_arr = {}

    def _process(self, vocab):
        if self.replace_zero:
            logger.debug('Replace digits with zero')
            vocab = _replace_zero(vocab)

        if self.lower:
            logger.debug('Lowercase')
            vocab = _lower(vocab)

        return vocab

    @staticmethod
    def _stem(filename):
        '''
        vocab.words.txt -> word
        '''
        _, attr, _ = filename.split('.')
        return attr[:-1]

    def _compile(self):
        for filename, vocab in self.vocab_arr.items():
            if filename in ['vocab.words.txt', 'vocab.chars.txt']:
                vocab = self._process(vocab)
            vocab_arr = sorted(list(set(vocab)))
            self.vocab_arr[filename] = vocab_arr

        for filename, vocab_arr in self.vocab_arr.items():
            dictionary = {w: i for i, w in enumerate(vocab_arr)}
            name = self._stem(filename)
            self.dictionaries[f'{name}2idx'] = dictionary

    def _init_vocab(self, vocab_file):
        vocab_file_name = vocab_file.name
        with open(vocab_file, encoding='utf-8') as vocab_file:
            vocab_txt = vocab_file.read()
            vocab_txt = vocab_txt.rstrip('\n')
            vocab_arr = vocab_txt.split('\n')
            self.vocab_arr[vocab_file_name] = vocab_arr

    def _update_vocab(self, vocab_file):
        vocab_file_name = vocab_file.name
        with open(vocab_file, encoding='utf-8') as vocab_file:
            vocab_txt = vocab_file.read()
            vocab_txt = vocab_txt.rstrip('\n')
            vocab_arr = vocab_txt.split('\n')
            self.vocab_arr[vocab_file_name] += vocab_arr
            logger.debug(f'Updated the dictionary using {vocab_file}')

    def _load_vocab(self, elem_name):
        vocab_path = self.data_path / f'vocab.{elem_name}.txt'
        if vocab_path.exists():
            self._init_vocab(vocab_path)

        if f'vocab.{elem_name}.txt' in self.params:
            logger.debug(f'Use additional vocabulary for {elem_name}')
            additional_vocab_path = Path(self.params[f'vocab.{elem_name}.txt'])
            self._update_vocab(additional_vocab_path)

    @classmethod
    def prepare(cls, params):
        vocab = cls(params)

        elem_names = ['words', 'chars', 'tags', 'labels']
        for elem_name in elem_names:
            vocab._load_vocab(elem_name)

        vocab._compile()
        logger.debug('Built vocabulary')
        return vocab

    def load_word_sentences(self, file):
        sentences = []
        with open(file, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip('\n')
                words = line.split(' ')

                if self.replace_zero:
                    words = _replace_zero(words)
                if self.lower:
                    words = _lower(words)

                sentences.append(words)
        return sentences

    def load_tag_sentences(self, file):
        sentences = []
        with open(file, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip('\n')
                sentences.append(line.split(' '))
        return sentences
