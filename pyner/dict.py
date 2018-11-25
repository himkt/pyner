from pathlib import Path

import logging
import numpy


logger = logging.getLogger(__name__)


class Dictionary:
    def __init__(self, params):
        self.__name__ = 'Dictionary'
        self.word2idx = {}
        self.label2idx = {}
        self.wordid2labelid = {}
        self.n_ignore = 0

        self.params = params
        self.data_path = Path(params['dictionary_base_path'])

    def build_dictionary_matrix(self):
        n_word_vocab = len(self.word2idx)
        n_label_vocab = len(self.label2idx)
        label_matrix = numpy.zeros((n_word_vocab, n_label_vocab))
        for wordid, labelid in self.wordid2labelid.items():
            label_matrix[wordid, labelid] = 1
        return label_matrix

    def _update_dictionary(self, name):
        words_path = self.data_path / f'{name}.words.txt'
        labels_path = self.data_path / f'{name}.labels.txt'
        words_file = open(words_path, encoding='utf-8')
        labels_file = open(labels_path, encoding='utf=8')

        for word, label in zip(words_file, labels_file):
            word = word.rstrip('\n')
            label = label.rstrip('\n')

            if word not in self.word2idx:
                self.n_ignore += 1
                continue

            wordid = self.word2idx[word]
            self.word2idx[word] = wordid

            labelid = self.label2idx.get(label, len(self.label2idx))
            self.label2idx[label] = labelid
            self.wordid2labelid[wordid] = labelid

        words_file.close()
        labels_file.close()

    @classmethod
    def prepare(cls, params, word2idx):
        dictionary = cls(params)
        dictionary.word2idx = word2idx

        role_names = ['train', 'validation', 'test']
        for name in role_names:
            dictionary._update_dictionary(name)

        logger.debug(f'{dictionary.n_ignore} words ignored')
        return dictionary
