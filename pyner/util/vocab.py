from pathlib import Path

import logging
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


def _replace_zero(ws):
    ws = [re.sub(r'\d', '0', w) for w in ws]
    return ws


class Vocabulary:
    def __init__(self, params):
        self.__name__ = 'Vocabulary'

        self.replace_zero = params.get('replace_zero', False)
        self.lower = params.get('lower', False)

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

                sentences.append(words)
        return sentences

    def load_tag_sentences(self, file):
        sentences = []
        with open(file, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip('\n')
                sentences.append(line.split(' '))
        return sentences
