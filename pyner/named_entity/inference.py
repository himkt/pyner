from pyner.named_entity.dataset import converter
from pyner.named_entity.dataset import DatasetTransformer
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util.argparse import parse_inference_args
from pyner.util.deterministic import set_seed
from pyner.util.iterator import create_iterator
from pyner.util.vocab import Vocabulary
from pyner.util.metric import select_snapshot

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
    configs = json.load(open(model_dir / 'args'))

    metric = args.metric.replace('/', '.')
    snapshot_file, prediction_path = select_snapshot(args, model_dir)
    logger.debug(f'creat prediction into {prediction_path}')

    vocab = Vocabulary.prepare(configs)
    num_word_vocab = configs['num_word_vocab']
    num_char_vocab = configs['num_char_vocab']
    num_tag_vocab = configs['num_tag_vocab']

    model = BiLSTM_CRF(
        configs,
        num_word_vocab,
        num_char_vocab,
        num_tag_vocab
    )

    model_path = model_dir / snapshot_file
    logger.debug(f'load {snapshot_file}')
    chainer.serializers.load_npz(model_path.as_posix(), model)

    if args.device >= 0:
        model.to_gpu(args.device)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform
    test_iterator = create_iterator(vocab, configs, 'test', transform)

    with open(prediction_path, 'w', encoding='utf-8') as file:
        for batch in test_iterator:
            in_arrays, t_arrays = converter(batch, args.device)
            p_arrays = model.predict(in_arrays)

            word_sentences, t_tag_sentences = list(zip(*transformer.itransform(
                in_arrays[0], t_arrays)))
            _, p_tag_sentences = list(zip(*transformer.itransform(
                in_arrays[0], p_arrays)))

            sentence_gen = zip(word_sentences, t_tag_sentences, p_tag_sentences)  # NOQA
            for ws, ts, ps in sentence_gen:
                for w, t, p in zip(ws, ts, ps):
                    print(f'{w} {t} {p}', file=file)
                print(file=file)
