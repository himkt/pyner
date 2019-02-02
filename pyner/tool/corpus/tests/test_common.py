from corpus.common import CorpusParser
from corpus.common import enum

import unittest


TEST_CoNLL2003_1 = '''\
-DOCSTART- -X- O O
U.N.         NNP  I-NP  I-ORG 
official     NN   I-NP  O 
Ekeus        NNP  I-NP  I-PER 
heads        VBZ  I-VP  O 
for          IN   I-PP  O 
Baghdad      NNP  I-NP  I-LOC 
.            .    O     O 
'''.split('\n')  # NOQA


WORDS_CoNLL_1 = 'U.N. official Ekeus heads for Baghdad .'.split(' ')  # NOQA
TAGS_CoNLL_1 = 'I-ORG O I-PER O O I-LOC O'.split(' ')  # NOQA


class TestEnumerate(unittest.TestCase):
    def test_enum1(self):
        word_sentences = [['今日', 'の', 'ごはん', 'は', 'カレー', 'カレー', '！！']]
        tag_sentences = [['O', 'O', 'B-F', 'O', 'B-F']]
        ws, cs, ts = enum(word_sentences, tag_sentences)
        expected_ws = sorted(['今日', 'の', 'ごはん', 'は', 'カレー', '！！'])  # NOQA
        expected_cs = sorted(list('今日のごはんカレー！'))
        expected_ts = sorted(['O', 'B-F'])

        self.assertEqual(ws, expected_ws)
        self.assertEqual(cs, expected_cs)
        self.assertEqual(ts, expected_ts)


class TestParser(unittest.TestCase):
    def test_parse1(self):
        parser = CorpusParser()
        word_sentences, tag_sentences = \
            parser._parse(TEST_CoNLL2003_1, word_idx=0, tag_idx=-1)
        print(word_sentences)
        self.assertEqual(word_sentences, [WORDS_CoNLL_1])
        self.assertEqual(tag_sentences, [TAGS_CoNLL_1])
