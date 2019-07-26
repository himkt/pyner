from pyner.word_class.dataset import converter
from pyner.word_class.dataset import DatasetTransformer
from pyner.word_class.dataset import WordClassDataset
from pyner.word_class.classifier import CharacterBasedClassifier
from pyner.vocab import Vocabulary
from pyner.util import parse_inference_args
from pyner.util import select_snapshot
from pyner.util import set_seed
from pathlib import Path

import chainer.iterators as It

import chainer
import logging
import json
import sys


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    fmt = '%(asctime)s : %(threadName)s : %(levelname)s : %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    args = parse_inference_args()
    chainer.config.train = False

    if args.device >= 0:
        chainer.cuda.get_device(args.device).use()
    set_seed()

    model_dir = Path(args.model)
    params = json.load(open(model_dir / 'args'))

    snapshot_file, prediction_path = select_snapshot(args, model_dir)
    logger.debug(f'creat prediction into {prediction_path}')

    model = CharacterBasedClassifier(params)
    model_path = model_dir / snapshot_file
    print(f'load {snapshot_file}', file=sys.stderr)
    chainer.serializers.load_npz(model_path.as_posix(), model)

    if args.device >= 0:
        model.to_gpu(args.device)

    logger.debug('*** parameters ***')
    for key, value in params.items():
        logger.debug(f'{key}: {value}')

    vocab = Vocabulary.prepare(params)
    transformer = DatasetTransformer(vocab)

    test_dataset = WordClassDataset(params, 'test',
                                    transformer.transform)

    test_iterator = It.SerialIterator(test_dataset,
                                      batch_size=len(test_dataset),
                                      repeat=False,
                                      shuffle=False)

    with open(prediction_path, 'w', encoding='utf-8') as file:
        for batch in test_iterator:
            in_arrays, t_arrays = converter(batch, args.device)
            p_arrays = model.predict(in_arrays)
            word_array, ts = list(zip(*transformer.itransform(
                in_arrays, t_arrays)))
            _, ps = list(zip(*transformer.itransform(
                in_arrays, p_arrays)))

            for w, t, p in zip(word_array, ts, ps):
                print(f'{w} {t} {p}', file=file)
