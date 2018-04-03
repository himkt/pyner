from chainerui.utils import save_args

import converters
import evaluators
import networks
import parsers

import argparse
import chainer
import pathlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='CoNLL format labeled data for training')  # NOQA
    # NOTE currently we do not use develop dataset for any perpose
    parser.add_argument('--develop', help='CoNLL format labeled data for tuning')  # NOQA
    parser.add_argument('--test', help='CoNLL format labeled data for test')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', type=str, default='neural_crf')
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--word-vec', type=str)
    parser.add_argument('--char-dim', type=int, default=25)
    parser.add_argument('--word-dim', type=int, default=200)
    parser.add_argument('--fix-embedding', action='store_true')
    parser.add_argument('--task', type=str, default='ner', help='Specify the task which you solve [chk/ner]')  # NOQA
    args = parser.parse_args()

    assert args.task in ['chk', 'ner'], "invalid task"
    target_tag = f'{args.task}tag'

    name = f'{args.model}_{args.task}'
    result_path = pathlib.PurePath(args.result_dir, name)
    result_fpath = result_path.as_posix()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    conll_parser = parsers.CoNLLParser()
    conll_parser.read_data(args.train)

    train_chars = conll_parser.get_data(args.train, 'char')  # NOQA
    train_words = conll_parser.get_data(args.train, 'word')  # NOQA
    train_postags = conll_parser.get_data(args.train, 'postag')  # NOQA
    train_labels = conll_parser.get_data(args.train, target_tag)  # NOQA
    train_dataset = chainer.datasets.TupleDataset(train_words, train_chars, train_postags, train_labels)  # NOQA
    train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size=32)  # NOQA
    n_train_vocab = len(conll_parser.word2index)

    if args.develop:
        conll_parser.read_data(args.develop)
        dev_chars = conll_parser.get_data(args.dev_label_data, 'char')  # NOQA
        dev_words = conll_parser.get_data(args.dev_label_data, 'word')  # NOQA
        dev_postags = conll_parser.get_data(args.dev_label_data, 'postag')  # NOQA
        dev_labels = conll_parser.get_data(args.dev_label_data, target_tag)  # NOQA
        dev_dataset = chainer.datasets.TupleDataset(dev_words, dev_chars, dev_postags, dev_labels)  # NOQA
        dev_iter = chainer.iterators.SerialIterator(dev_dataset, batch_size=1024, repeat=False)  # NOQA

    if args.test:
        conll_parser.read_data(args.test)
        test_chars = conll_parser.get_data(args.test, 'char')  # NOQA
        test_words = conll_parser.get_data(args.test, 'word')  # NOQA
        test_postags = conll_parser.get_data(args.test, 'postag')  # NOQA
        test_labels = conll_parser.get_data(args.test, target_tag)  # NOQA
        test_dataset = chainer.datasets.TupleDataset(test_words, test_chars, test_postags, test_labels)  # NOQA
        test_iter = chainer.iterators.SerialIterator(test_dataset, batch_size=1024, repeat=False)  # NOQA

    syn0 = None
    if args.word_vec:
        syn0 = conll_parser.load_embedding(args.word_vec)

    word2index = conll_parser.word2index
    char2index = conll_parser.char2index
    postag2index = conll_parser.postag2index

    if args.task == 'ner':
        label2index = conll_parser.nertag2index

    else:
        label2index = conll_parser.chktag2index

    n_vocab, n_char, n_postag, n_label = len(word2index), len(char2index), len(postag2index), len(label2index)  # NOQA
    postag_dim, hidden_dim, char_hidden_dim, fw_dim, bw_dim = 15, 100, 100, 30, 30  # NOQA

    word_dim = args.word_dim
    char_dim = args.char_dim

    if syn0 is not None:
        print('use pre-trained embedding')

    if args.model == 'crf':
        nn = networks.CRF(n_vocab, n_postag, n_label, word_dim, postag_dim, syn0)  # NOQA
        converter = converters.converter

    elif args.model == 'bilstm':
        nn = networks.BiLSTM(n_vocab, n_postag, n_label, word_dim, postag_dim, hidden_dim, syn0)  # NOQA
        converter = converters.converter

    elif args.model == 'bilstm_crf':
        nn = networks.BiLSTM_CRF(n_vocab, n_postag, n_label, word_dim, postag_dim, hidden_dim, syn0)  # NOQA
        converter = converters.converter

    elif args.model == 'char_bilstm_crf':
        nn = networks.Char_BiLSTM_CRF(n_vocab, n_char, n_postag, n_label, word_dim, char_dim, postag_dim,  # NOQA
                                      hidden_dim, char_hidden_dim, syn0)
        converter = converters.converter2

    elif args.model == 'semi_bilstm_crf':
        nn = networks.Semi_BiLSTM_CRF(n_vocab, n_postag, n_train_vocab, n_label,  # NOQA
                                      word_dim, char_dim, postag_dim, fw_dim, bw_dim, syn0)  # NOQA
        converter = converters.converter

    elif args.model == 'semi_bilstm_crf2':
        nn = networks.Semi_BiLSTM_CRF2(n_vocab, n_postag, n_train_vocab, n_label,  # NOQA
                                       word_dim, char_dim, postag_dim, fw_dim, bw_dim, syn0)  # NOQA
        converter = converters.converter

    print('model: ', nn)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(nn)

    if args.fix_embedding:
        print('Fix embedding')
        nn.embed_word.disable_update()

    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=converter, device=args.gpu)  # NOQA
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=result_fpath)  # NOQA
    save_args(args, result_fpath)

    if args.test:
        trainer.extend(chainer.training.extensions.Evaluator(test_iter, nn, converter=converter, device=args.gpu))  # NOQA
        trainer.extend(evaluators.NEREvaluator(test_iter, nn, converter, args.gpu, label2index))  # NOQA

    trainer.extend(chainer.training.extensions.ProgressBar())
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.run()
