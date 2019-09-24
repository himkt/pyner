from pyner.tool.corpus.common import CorpusParser
from pyner.tool.corpus.common import bio2bioes
from pyner.tool.corpus.common import enum
from pyner.tool.corpus.common import iob2bio

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
TEST_IOB1 = ['I-PER', 'I-ORG', 'I-ORG', 'B-ORG', 'O']
TEST_BIO1 = ['B-PER', 'B-ORG', 'I-ORG', 'B-ORG', 'O']
TEST_BIOES1 = ['S-PER', 'B-ORG', 'E-ORG', 'S-ORG', 'O']


def test_enum1():
    word_sentences = [['今日', 'の', 'ごはん', 'は', 'カレー', 'カレー', '！！']]
    tag_sentences = [['O', 'O', 'B-F', 'O', 'B-F']]
    ws, cs, ts = enum(word_sentences, tag_sentences)
    expected_ws = sorted(['今日', 'の', 'ごはん', 'は', 'カレー', '！！'])  # NOQA
    expected_cs = sorted(list('今日のごはんカレー！'))
    expected_ts = sorted(['O', 'B-F'])

    assert ws == expected_ws
    assert cs == expected_cs
    assert ts == expected_ts


def test_parse1():
    parser = CorpusParser(convert_rule=None, delimiter=r" +")
    word_sentences, tag_sentences = parser._parse(
        TEST_CoNLL2003_1,
        word_idx=0,
        tag_idx=-1)
    assert word_sentences == [WORDS_CoNLL_1]
    assert tag_sentences == [TAGS_CoNLL_1]


def test_iob2bio1():
    assert iob2bio(TEST_IOB1) == TEST_BIO1


def test_bio2bioes1():
    assert bio2bioes(iob2bio(TEST_BIO1)) == TEST_BIOES1


def test_iob2bioes1():
    assert bio2bioes(iob2bio(TEST_IOB1)) == TEST_BIOES1
