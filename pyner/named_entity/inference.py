import json
import logging
import pathlib

import chainer
import click

from pyner.named_entity.dataset import DatasetTransformer, converter
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util.deterministic import set_seed
from pyner.util.iterator import create_iterator
from pyner.util.metric import select_snapshot
from pyner.util.vocab import Vocabulary


@click.command()
@click.argument("model")
@click.option("--epoch", type=int, required=True)
@click.option("--device", type=int, default=-1)
@click.option("--metric", type=str, default="validation/main/fscore")
def run_inference(model: str, epoch: int, device: str, metric: str):
    chainer.config.train = False

    if device >= 0:
        chainer.cuda.get_device(device).use()

    set_seed()

    model_dir = pathlib.Path(model)
    configs = json.load(open(model_dir / "args"))

    metric = metric.replace("/", ".")
    snapshot_file, prediction_path = select_snapshot(
        epoch, metric, model, model_dir)
    logger.debug(f"creat prediction into {prediction_path}")

    vocab = Vocabulary.prepare(configs)
    num_word_vocab = configs["num_word_vocab"]
    num_char_vocab = configs["num_char_vocab"]
    num_tag_vocab = configs["num_tag_vocab"]

    model = BiLSTM_CRF(configs, num_word_vocab, num_char_vocab, num_tag_vocab)

    model_path = model_dir / snapshot_file
    logger.debug(f"load {snapshot_file}")
    chainer.serializers.load_npz(model_path.as_posix(), model)

    if device >= 0:
        model.to_gpu(device)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform
    test_iterator = create_iterator(vocab, configs, "test", transform)

    with open(prediction_path, "w", encoding="utf-8") as file:
        for batch in test_iterator:
            in_arrays, t_arrays = converter(batch, device)
            p_arrays = model.predict(in_arrays)

            word_sentences, t_tag_sentences = list(
                zip(*transformer.itransform(in_arrays[0], t_arrays))
            )
            _, p_tag_sentences = list(
                zip(*transformer.itransform(in_arrays[0], p_arrays))
            )

            sentence_gen = zip(word_sentences, t_tag_sentences, p_tag_sentences)  # NOQA
            for ws, ts, ps in sentence_gen:
                for w, t, p in zip(ws, ts, ps):
                    print(f"{w} {t} {p}", file=file)
                print(file=file)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    fmt = "%(asctime)s : %(threadName)s : %(levelname)s : %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    run_inference()
