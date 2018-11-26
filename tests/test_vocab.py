from pyner.vocab import Vocabulary

import unittest


params1 = {'vocab.words.txt': './tests/test_data/v1/vocab.words.txt',
           'data_dir': './tests/test_data/v2'}

params2 = {'vocab.words.txt': './tests/test_data/v2/vocab.words.txt',
           'data_dir': './tests/test_data/v1'}

params3 = {'data_dir': './tests/test_data/v1',
           'replace_zero': True}

params4 = {'data_dir': './tests/test_data/v1',
           'lower': True}

params5 = {'data_dir': './tests/test_data/v1',
           'replace_zero': True, 'lower': True}


class VocabularyTest(unittest.TestCase):
    def test_merge_vocab(self):
        """
        Check if the order of loading vocab doesn't affect
        """
        vocab1 = Vocabulary.prepare(params1)
        vocab2 = Vocabulary.prepare(params2)

        result1 = vocab1.vocab_arr['vocab.words.txt']
        result2 = vocab2.vocab_arr['vocab.words.txt']
        self.assertEqual(result1, result2)

    def test_digit_normalize(self):
        vocab = Vocabulary.prepare(params3)
        result = vocab.vocab_arr['vocab.words.txt']
        expect = sorted(['apple00', 'wine', 'Apple00', 'Apple', 'apple'])
        self.assertEqual(expect, result)

    def test_case_normalize(self):
        vocab = Vocabulary.prepare(params4)
        result = vocab.vocab_arr['vocab.words.txt']
        expect = sorted(['apple01', 'wine', 'apple'])
        self.assertEqual(expect, result)

    def test_all_normalize(self):
        vocab = Vocabulary.prepare(params5)
        result = vocab.vocab_arr['vocab.words.txt']
        expect = sorted(['apple00', 'wine', 'apple'])
        self.assertEqual(expect, result)
