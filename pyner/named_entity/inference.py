import json
import logging
from typing import Optional

import click
import chainer
import os.path

from chainer.backends import cuda
from pyner.named_entity.dataset import DatasetTransformer, converter
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util.deterministic import set_seed
from pyner.util.iterator import create_iterator
from pyner.util.metric import select_snapshot
from pyner.util.vocab import Vocabulary


@click.command()
@click.argument("model_dir")
@click.option("--epoch", type=int)
@click.option("--device", type=int, default=-1)
@click.option("--metric", type=str, default="validation/main/fscore")
def run_inference(
        model_dir: str, epoch: Optional[int], device: str, metric: str):
    chainer.config.train = False

    if device >= 0:
        cuda.get_device(device).use()

    set_seed()

    configs = json.load(open(os.path.join(model_dir, "args")))
    snapshot_file, prediction_path = select_snapshot(epoch, metric, model_dir)
    logger.debug(f"creat prediction into {prediction_path}")

    vocab = Vocabulary.prepare(configs)
    num_word_vocab = configs["num_word_vocab"]
    num_char_vocab = configs["num_char_vocab"]
    num_tag_vocab = configs["num_tag_vocab"]

    model = BiLSTM_CRF(configs, num_word_vocab, num_char_vocab, num_tag_vocab)

    model_path = os.path.join(model_dir, snapshot_file)
    chainer.serializers.load_npz(model_path, model)
    logger.debug(f"load {snapshot_file}")

    if device >= 0:
        model.to_gpu(device)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform
    test_iterator = create_iterator(
        vocab,
        configs,
        "test",
        transform,
        return_original_sentence=True
    )

    with open(prediction_path, "w", encoding="utf-8") as file:
        for batch in test_iterator:
            batch, original_sentences = list(zip(*batch))
            in_arrays, t_arrays = converter(batch, device)
            p_arrays = model.predict(in_arrays)

            word_sentences, t_tag_sentences = list(
                zip(*transformer.itransform(in_arrays[0], t_arrays))
            )
            _, p_tag_sentences = list(
                zip(*transformer.itransform(in_arrays[0], p_arrays))
            )

            sentence_gen = zip(
                word_sentences,
                t_tag_sentences,
                p_tag_sentences,
                original_sentences,
            )  # NOQA
            for ws, ts, ps, _os in sentence_gen:
                for w, t, p, o in zip(ws, ts, ps, _os):
                    w = w.replace(" ", "<WHITESPACE>")
                    o = o.replace(" ", "<WHITESPACE>")
                    if w != o:
                        w = f"{w}({o})"
                    print(f"{w} {t} {p}", file=file)
                print(file=file)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    fmt = "%(asctime)s : %(threadName)s : %(levelname)s : %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    # pylint: disable=no-value-for-parameter
    run_inference()
