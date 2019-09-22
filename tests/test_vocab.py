from pyner.util.vocab import Vocabulary


PARAMS1 = {'external': {'data_dir': './tests/test_data/v1'},
           'preprocessing': {'replace_zero': True}}

PARAMS2 = {'external': {'data_dir': './tests/test_data/v1'},
           'preprocessing': {'lower': True}}

PARAMS3 = {'external': {'data_dir': './tests/test_data/v1'},
           'preprocessing': {'replace_zero': True, 'lower': True}}


def test_digit_normalize():
    vocab = Vocabulary.prepare(PARAMS1)
    result = vocab.vocab_arr['word']
    expect = sorted(['apple00', 'wine', 'Apple00', 'Apple', 'apple'])
    print('result: ', result)
    assert expect == result


def test_case_normalize():
    vocab = Vocabulary.prepare(PARAMS2)
    result = vocab.vocab_arr['word']
    expect = sorted(['apple01', 'wine', 'apple02', 'apple'])
    print('result: ', result)
    assert expect == result


def test_all_normalize():
    vocab = Vocabulary.prepare(PARAMS3)
    result = vocab.vocab_arr['word']
    expect = sorted(['apple00', 'wine', 'apple'])
    print('result: ', result)
    assert expect == result
