from pyner.dict import Dictionary
from pyner.extension import LearningRateDecay
from pyner.vocab import Vocabulary
from pyner.named_entity.dataset import converter
from pyner.named_entity.dataset import DatasetTransformer
from pyner.named_entity.dataset import SequenceLabelingDataset
from pyner.named_entity.evaluator import NamedEntityEvaluator
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util import add_hooks
from pyner.util import set_seed
from pyner.util import create_optimizer
from pyner.util import parse_train_args

from chainerui.utils import save_args
from pathlib import Path

import chainer.iterators as It
import chainer.training as T
import chainer.training.extensions as E

import chainer
import logging
import yaml


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    fmt = '%(asctime)s : %(threadName)s : %(levelname)s : %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)

    args = parse_train_args()
    params = yaml.load(open(args.config, encoding='utf-8'))

    if args.device >= 0:
        chainer.cuda.get_device(args.device).use()
    set_seed(args.seed, args.device)

    config_path = Path(args.config)
    model_path = Path(args.config.replace('config', 'model', 1))

    logger.debug(f'model_dir: {model_path}')
    vocab = Vocabulary.prepare(params)

    params['n_word_vocab'] = max(vocab.dictionaries['word2idx'].values()) + 1
    params['n_char_vocab'] = max(vocab.dictionaries['char2idx'].values()) + 1
    params['n_tag_vocab'] = max(vocab.dictionaries['tag2idx'].values()) + 1
    params['lower'] = vocab.lower

    word2idx = None
    label_matrix = None

    if 'word_vector' in params:
        word2idx = vocab.dictionaries['word2idx']

    if 'dictionary_base_path' in params:
        params['dictionary'] = True
        word2idx = vocab.dictionaries['word2idx']
        dictionary = Dictionary.prepare(params, word2idx)
        label_matrix = dictionary.build_dictionary_matrix()
        _, n_label_vocab = label_matrix.shape
        params['n_label_vocab'] = n_label_vocab

    model = BiLSTM_CRF(params, word2idx, label_matrix)
    if args.device >= 0:
        model.to_gpu(args.device)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform

    train_dataset = SequenceLabelingDataset(vocab, params, 'train', transform)
    valid_dataset = SequenceLabelingDataset(vocab, params, 'validation', transform)  # NOQA
    test_dataset = SequenceLabelingDataset(vocab, params, 'test', transform)

    train_iterator = It.SerialIterator(train_dataset,
                                       batch_size=params['batch_size'],
                                       shuffle=True)

    valid_iterator = It.SerialIterator(valid_dataset,
                                       batch_size=len(valid_dataset),
                                       shuffle=False,
                                       repeat=False)

    test_iterator = It.SerialIterator(test_dataset,
                                      batch_size=len(test_dataset),
                                      shuffle=False,
                                      repeat=False)

    optimizer = create_optimizer(params)
    optimizer.setup(model)
    optimizer = add_hooks(optimizer, params)

    updater = T.StandardUpdater(train_iterator, optimizer,
                                converter=converter,
                                device=args.device)

    save_args(params, model_path)
    trainer = T.Trainer(updater, (params['epoch'], 'epoch'), out=model_path)
    logger.debug(f'Create {model_path} for trainer\'s output')

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
