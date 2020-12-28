import json
import logging
from typing import Optional

import sys
import click
import chainer
import os

from tiny_tokenizer import WordTokenizer
from pyner.named_entity.dataset import DatasetTransformer, converter
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util.deterministic import set_seed
from pyner.util.metric import select_snapshot
from pyner.util.vocab import Vocabulary


@click.command()
@click.argument("model_dir")
@click.option("--epoch", type=int)
@click.option("--device", type=int, default=-1)
@click.option("--metric", type=str, default="validation/main/fscore")
@click.option("--tokenizer", type=str, default="mecab")
def run_inference(
    model_dir: str, epoch: Optional[int], device: str, metric: str, tokenizer: str
):
    chainer.config.train = False

    if device >= 0:
        chainer.get_device(device).use()

    set_seed()

    config = json.load(open(os.path.join(model_dir, "args")))
    snapshot_file, prediction_path = select_snapshot(epoch, metric, model_dir)
    logger.debug(f"creat prediction into {prediction_path}")

    vocab = Vocabulary.prepare(config)
    num_word_vocab = config["num_word_vocab"]
    num_char_vocab = config["num_char_vocab"]
    num_tag_vocab = config["num_tag_vocab"]

    model = BiLSTM_CRF(config, num_word_vocab, num_char_vocab, num_tag_vocab)

    model_path = os.path.join(model_dir, snapshot_file)
    logger.debug(f"load {snapshot_file}")
    chainer.serializers.load_npz(model_path, model)

    if device >= 0:
        model.to_gpu(device)

    transformer = DatasetTransformer(vocab)
    word_tokenizer = WordTokenizer(tokenizer=tokenizer)
    print("Successfully loading a model, please input a sentence")

    for line in sys.stdin:
        input_sentence = [str(t) for t in word_tokenizer.tokenize(line)]
        batch = transformer.transform(input_sentence, None)
        in_arr, _ = converter([batch])
        pd_arr = model.predict(in_arr)
        ((_, tag_sequence),) = transformer.itransform(in_arr[0], pd_arr)
        print(
            " ".join(f"{word}/{tag}" for word, tag in zip(input_sentence, tag_sequence))
        )  # NOQA


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    fmt = "%(asctime)s : %(threadName)s : %(levelname)s : %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    # pylint: disable=no-value-for-parameter
    run_inference()
