from pyner.util.vocab import Vocabulary

import unittest


params1 = {'external': {'data_dir': './tests/test_data/v1'},
           'preprocessing': {'replace_zero': True}}

params2 = {'external': {'data_dir': './tests/test_data/v1'},
           'preprocessing': {'lower': True}}

params3 = {'external': {'data_dir': './tests/test_data/v1'},
           'preprocessing': {'replace_zero': True, 'lower': True}}


class VocabularyTest(unittest.TestCase):
    def test_digit_normalize(self):
        vocab = Vocabulary.prepare(params1)
        result = vocab.vocab_arr['word']
        expect = sorted(['apple00', 'wine', 'Apple00', 'Apple', 'apple'])
        print('result: ', result)
        self.assertEqual(expect, result)

    def test_case_normalize(self):
        vocab = Vocabulary.prepare(params2)
        result = vocab.vocab_arr['word']
        expect = sorted(['apple01', 'wine', 'apple02', 'apple'])
        print('result: ', result)
        self.assertEqual(expect, result)

    def test_all_normalize(self):
        vocab = Vocabulary.prepare(params3)
        result = vocab.vocab_arr['word']
        expect = sorted(['apple00', 'wine', 'apple'])
        print('result: ', result)
        self.assertEqual(expect, result)
