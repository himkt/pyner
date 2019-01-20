from corpus.tag_scheme import bio2bioes
from corpus.tag_scheme import iob2bio

import unittest


TEST_IOB1 = ['I-PER', 'I-ORG', 'I-ORG', 'B-ORG', 'O']
TEST_BIO1 = ['B-PER', 'B-ORG', 'I-ORG', 'B-ORG', 'O']
TEST_BIOES1 = ['S-PER', 'B-ORG', 'E-ORG', 'S-ORG', 'O']


class TestEnumerate(unittest.TestCase):
    def test_iob2bio1(self):
        self.assertEqual(iob2bio(TEST_IOB1), TEST_BIO1)

    def test_bio2bioes1(self):
        self.assertEqual(bio2bioes(iob2bio(TEST_BIO1)), TEST_BIOES1)

    def test_iob2bioes1(self):
        self.assertEqual(bio2bioes(iob2bio(TEST_IOB1)), TEST_BIOES1)
