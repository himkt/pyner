from pyner import Vocabulary
from pyner.util import add_hooks
from pyner.util import create_optimizer
from pyner.util import set_seed
from pyner.util import parse_train_args
from pyner.word_class.evaluator import ClassificationEvaluator
from pyner.word_class.classifier import CharacterBasedClassifier
from pyner.word_class.dataset import DatasetTransformer
from pyner.word_class.dataset import WordClassDataset
from pyner.word_class.dataset import converter

from chainerui.utils import save_args
from pathlib import Path

import chainer.iterators as It
import chainer.training as T
import chainer.training.extensions as E

import chainer
import logging
import yaml


if __name__ == '__main__':
    chainer.global_config.cudnn_deterministic = True
    logger = logging.getLogger(__name__)
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)

    args = parse_train_args()
    params = yaml.load(open(args.config))

    if args.device >= 0:
        chainer.cuda.get_device(args.device).use()
    set_seed(args.seed, args.device)

    data_path = Path(params['data_dir'])
    config_path = Path(args.config)
    model_path = Path(args.config.replace('config', 'model', 1))

    logger.debug(f'model_dir: {model_path}')
    vocab = Vocabulary.prepare(params)

    max_charid = max(vocab.dictionaries['char2idx'].values())
    max_labelid = max(vocab.dictionaries['label2idx'].values())
    params['n_char_vocab'] = max_charid + 1
    params['n_label_vocab'] = max_labelid + 1

    model = CharacterBasedClassifier(params)

    if args.device >= 0:
        model.to_gpu(args.device)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform

    train_dataset = WordClassDataset(params, 'train', transform)
    valid_dataset = WordClassDataset(params, 'validation', transform)

    train_iterator = It.MultiprocessIterator(train_dataset,
                                             batch_size=params['batch_size'],
                                             n_processes=8,
                                             n_prefetch=10,
                                             shared_mem=1001001,
                                             shuffle=True)

    valid_iterator = It.MultiprocessIterator(valid_dataset,
                                             batch_size=len(valid_dataset),
                                             n_processes=8,
                                             n_prefetch=10,
                                             shared_mem=1001001,
                                             repeat=False,
                                             shuffle=False)

    optimizer = create_optimizer(params)
    optimizer.setup(model)
    optimizer = add_hooks(optimizer, params)

    updater = T.StandardUpdater(train_iterator, optimizer,
                                converter=converter,
                                device=args.device)

    save_args(params, model_path)
    trainer = T.Trainer(updater, (params['epoch'], 'epoch'), out=model_path)
    logger.debug(f'Create {model_path} for trainer\'s output')

    entries = ['epoch', 'iteration', 'elapsed_time',
               'main/loss', 'validation/main/loss',
               'validation/main/fscore']

    evaluator = ClassificationEvaluator(valid_iterator, model,
                                        transformer.itransform,
                                        converter, device=args.device)

    trainer.extend(evaluator)
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(entries=entries))
    trainer.extend(E.ProgressBar(update_interval=20))
    trainer.extend(E.snapshot_object(
        model, filename='snapshot_epoch_{.updater.epoch:04d}'),
                   trigger=(1, 'epoch'))

    trainer.run()
