from corpus.common import CorpusParser
from corpus.common import enum

import unittest


TEST_CoNLL2003_1 = '''
-DOCSTART- -X- O O

EU NNP I-NP I-ORG
rejects VBZ I-VP O
German JJ I-NP I-MISC
call NN I-NP O
to TO I-VP O
boycott VB I-VP O
British JJ I-NP I-MISC
lamb NN I-NP O
. . O O

Peter NNP I-NP I-PER
Blackburn NNP I-NP I-PER

BRUSSELS NNP I-NP I-LOC
1996-08-22 CD I-NP O
'''.split('\n')


TEST_rNE_1 = '''\
ID=00040625-proc.wordpart-001-01
001 002 白菜     名詞   F-B
002 014 を       助詞   O  
003 004 繊維     名詞   Sf-B
004 005 に       助詞   Sf-I
005 006 対       動詞   Sf-I
006 007 し       語尾   Sf-I
007 008 直角     名詞   Sf-I
008 009 に       助詞   O  
009 014 、       補助記号 O  
010 011 １       名詞   Sf-B
011 012 ／       補助記号 Sf-I
012 013 ３       名詞   Sf-I
013 014 に       助詞   O  
014 015 切り分け 動詞   Ac-B
015 016 る       語尾   O  
016  -1 。       補助記号 O  
'''.split('\n')


WORDS_rNE_1 = '白菜 を 繊維 に 対 し 直角 に 、 １ ／ ３ に 切り分け る 。'.split(' ')
# NOTE suffix style annotataion will be converted to prefix style one
TAGS_rNE_1 = 'B-F O B-Sf I-Sf I-Sf I-Sf I-Sf O O B-Sf I-Sf I-Sf O B-Ac O O'.split(' ')  # NOQA


SPECIAL_SYMBOLS = ['<PAD>', '<UNK>']
SPECIAL_SYMBOLS_TAG = ['<PAD>']


class TestEnumerate(unittest.TestCase):
    def test_enum1(self):
        word_sentences = [['今日', 'の', 'ごはん', 'は', 'カレー', 'カレー', '！！']]
        tag_sentences = [['O', 'O', 'B-F', 'O', 'B-F']]
        ws, cs, ts = enum(word_sentences, tag_sentences)
        expected_ws = SPECIAL_SYMBOLS + sorted(['今日', 'の', 'ごはん', 'は', 'カレー', '！！'])  # NOQA
        expected_cs = SPECIAL_SYMBOLS + sorted(list('今日のごはんカレー！'))
        expected_ts = SPECIAL_SYMBOLS_TAG + sorted(['O', 'B-F'])

        self.assertEqual(ws, expected_ws)
        self.assertEqual(cs, expected_cs)
        self.assertEqual(ts, expected_ts)


class TestParser(unittest.TestCase):
    def test_parse1(self):
        parser = CorpusParser()
        word_sentences, tag_sentences = \
            parser._parse(TEST_rNE_1, word_idx=2, tag_idx=-1)
        print(word_sentences)
        self.assertEqual(word_sentences, [WORDS_rNE_1])
        self.assertEqual(tag_sentences, [TAGS_rNE_1])
