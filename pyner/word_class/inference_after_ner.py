from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.word_class.dataset import converter
from pyner.word_class.dataset import DatasetTransformer
from pyner.word_class.dataset import WordClassDataset
from pyner.vocab import Vocabulary
from pyner.util import parse_inference_args
from pyner.util import select_snapshot
from pyner.util import set_seed

import chainer.iterators as It
import chainer
import pathlib
import logging
import json


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    fmt = '%(asctime)s : %(threadName)s : %(levelname)s : %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    args = parse_inference_args()
    chainer.config.train = False

    if args.device >= 0:
        chainer.cuda.get_device(args.device).use()
    set_seed()

    model_dir = pathlib.Path(args.model)
    params = json.load(open(model_dir / 'args'))

    vocab = Vocabulary.prepare(params)
    vocab.data_path = pathlib.Path('./data/CookingOntology')
    vocab._load_vocab('labels')
    vocab._compile()
    metric = args.metric.replace('/', '.')

    snapshot_file, prediction_path = select_snapshot(args, model_dir)
    logger.debug(f'creat prediction into {prediction_path}')

    model = BiLSTM_CRF(params)
    model_path = model_dir / snapshot_file
    logger.debug(f'load {snapshot_file}')
    chainer.serializers.load_npz(model_path.as_posix(), model)
    model = model.classifier

    if args.device >= 0:
        model.to_gpu(args.device)

    logger.debug('*** parameters ***')
    for key, value in params.items():
        logger.debug(f'{key}: {value}')

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform

    params['data_dir'] = './data/CookingOntology'
    test_dataset = WordClassDataset(params, 'test',
                                    transformer.transform)

    test_iterator = It.SerialIterator(test_dataset,
                                      batch_size=len(test_dataset),
                                      shuffle=False,
                                      repeat=False)

    with open('/dev/stdout', 'w', encoding='utf-8') as file:
        for batch in test_iterator:
            in_arrays, t_arrays = converter(batch, args.device)
            p_arrays = model(in_arrays).array.argmax(axis=1)
            word_array, ts = list(zip(*transformer.itransform(
                in_arrays, t_arrays)))
            _, ps = list(zip(*transformer.itransform(
                in_arrays, p_arrays)))

            for w, t, p in zip(word_array, ts, ps):
                print(f'{w} {t} {p}', file=file)
