from pyner.named_entity.dataset import converter
from pyner.named_entity.dataset import DatasetTransformer
from pyner.named_entity.dataset import SequenceLabelingDataset
from pyner.named_entity.evaluator import NamedEntityEvaluator
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util.argparse import parse_train_args
from pyner.util.config import ConfigParser
from pyner.util.deterministic import set_seed
from pyner.util.vocab import Vocabulary
from pyner.util.optimizer import create_optimizer
from pyner.util.optimizer import add_hooks
from pyner.util.optimizer import LearningRateDecay

from chainerui.utils import save_args
from pathlib import Path

import chainer.iterators as It
import chainer.training as T
import chainer.training.extensions as E

import chainer
import logging
import yaml


def prepare_pretrained_word_vector(
        word2idx,
        word_vector_path,
        syn0_original,
        lowercase=False
):

    import gensim
    import numpy

    num_word_vocab, word_dim = syn0_original.shape
    word_vector_gensim = gensim.models.KeyedVectors.load(
        word_vector_path
    )
    syn0 = numpy.zeros(
        [num_word_vocab, word_dim],
        dtype=numpy.float32
    )

    assert word_vector_gensim.wv.vector_size == word_dim

    # if lowercased word is in pre-trained embeddings,
    # increment match2
    match1, match2 = 0, 0

    for word, idx in word2idx.items():
        if word in word_vector_gensim:
            word_vector = word_vector_gensim.wv.word_vec(word)
            syn0[idx, :] = word_vector
            match1 += 1

        elif lowercase and word.lower() in word_vector_gensim:
            word_vector = word_vector_gensim.wv.word_vec(word.lower())
            syn0[idx, :] = word_vector
            match2 += 1

    match = match1 + match2
    matching_rate = 100 * (match/num_word_vocab)
    logger.info(f'Found {matching_rate:.2f}% words in pre-trained vocab')
    logger.info(f'- num_word_vocab: {num_word_vocab}')
    logger.info(f'- match1: {match1}, match2: {match2}')
    return syn0


def create_iterator(vocab, external_config, batch_config, role, transform):
    dataset = SequenceLabelingDataset(vocab, external_config, role, transform)
    shuffle = True if role == 'train' else False
    repeat = True if role == 'train' else False
    batch_size = batch_config['batch_size'] if role == 'train' else len(dataset)
    iterator = It.SerialIterator(
        dataset,
        batch_size=batch_size,
        repeat=repeat,
        shuffle=shuffle
    )
    return iterator


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    fmt = '%(asctime)s : %(threadName)s : %(levelname)s : %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)

    args = parse_train_args()
    params = yaml.load(open(args.config, encoding='utf-8'))

    if args.device >= 0:
        chainer.cuda.get_device(args.device).use()
    set_seed(args.seed, args.device)

    config_parser = ConfigParser(args.config)
    config_path = Path(args.config)

    batch_config = config_parser['batch']
    external_config = config_parser['external']
    model_config = config_parser['model']
    optimizer_config = config_parser['optimizer']
    model_path = Path(config_parser['output'])

    logger.debug(f'model_dir: {model_path}')
    vocab = Vocabulary.prepare(config_parser['external'])

    num_word_vocab = max(vocab.dictionaries['word2idx'].values()) + 1
    num_char_vocab = max(vocab.dictionaries['char2idx'].values()) + 1
    num_tag_vocab = max(vocab.dictionaries['tag2idx'].values()) + 1

    model = BiLSTM_CRF(model_config,
                       num_word_vocab,
                       num_char_vocab,
                       num_tag_vocab)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform

    if 'word_vector' in external_config:
        word2idx = vocab.dictionaries['word2idx']
        syn0 = prepare_pretrained_word_vector(
            word2idx,
            external_config['word_vector'],
            model.embed_word.W.data
        )
        model.set_pretrained_word_vectors(syn0)

    train_iterator = create_iterator(vocab, external_config,
                                     batch_config, 'train', transform)
    valid_iterator = create_iterator(vocab, external_config,
                                     batch_config, 'validation', transform)
    test_iterator = create_iterator(vocab, external_config,
                                    batch_config, 'test', transform)

    if args.device >= 0:
        model.to_gpu(args.device)

    optimizer = create_optimizer(optimizer_config)
    optimizer.setup(model)
    optimizer = add_hooks(optimizer, params)

    updater = T.StandardUpdater(train_iterator, optimizer,
                                converter=converter,
                                device=args.device)

    params = config_parser.export()
    params['num_word_vocab'] = num_word_vocab
    params['num_char_vocab'] = num_char_vocab
    params['num_tag_vocab'] = num_tag_vocab
    save_args(params, model_path)

    logger.debug(f'Create {model_path} for trainer\'s output')
    trigger = (batch_config['epoch'], 'epoch')

    trainer = T.Trainer(
        updater,
        trigger,
        out=model_path
    )

    entries = ['epoch', 'iteration', 'elapsed_time', 'lr',
               'main/loss', 'validation/main/loss',
               'validation/main/fscore',
               'validation_1/main/loss',
               'validation_1/main/fscore']

    valid_evaluator = NamedEntityEvaluator(valid_iterator, model,
                                           transformer.itransform,
                                           converter, device=args.device)

    test_evaluator = NamedEntityEvaluator(test_iterator, model,
                                          transformer.itransform,
                                          converter, device=args.device)

    epoch_trigger = (1, 'epoch')
    snapshot_filename = 'snapshot_epoch_{.updater.epoch:04d}'
    trainer.extend(valid_evaluator, trigger=epoch_trigger)
    trainer.extend(test_evaluator, trigger=epoch_trigger)
    trainer.extend(E.observe_lr(), trigger=epoch_trigger)
    trainer.extend(E.LogReport(trigger=epoch_trigger))
    trainer.extend(E.PrintReport(entries=entries), trigger=epoch_trigger)
    trainer.extend(E.ProgressBar(update_interval=20))
    trainer.extend(E.snapshot_object(model, filename=snapshot_filename),
                   trigger=(1, 'epoch'))

    if 'learning_rate_decay' in params:
        logger.debug('Enable Learning Rate decay')
        trainer.extend(LearningRateDecay('lr', params['learning_rate'],
                                         params['learning_rate_decay']),
                       trigger=epoch_trigger)

    trainer.run()
