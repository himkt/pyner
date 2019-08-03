import datetime
import logging
from pathlib import Path

import chainer
import chainer.training as T
import chainer.training.extensions as E
import click
import yaml
from chainerui.utils import save_args

from pyner.named_entity.dataset import DatasetTransformer, converter
from pyner.named_entity.evaluator import NamedEntityEvaluator
from pyner.named_entity.recognizer import BiLSTM_CRF
from pyner.util.config import ConfigParser
from pyner.util.deterministic import set_seed
from pyner.util.iterator import create_iterator
from pyner.util.optimizer import LearningRateDecay, add_hooks, create_optimizer
from pyner.util.vocab import Vocabulary


def prepare_pretrained_word_vector(
        word2idx, gensim_model, syn0, num_word_vocab):

    # if lowercased word is in pre-trained embeddings,
    # increment match2
    match1, match2 = 0, 0

    for word, idx in word2idx.items():
        if word in gensim_model:
            word_vector = gensim_model.wv.word_vec(word)
            syn0[idx, :] = word_vector
            match1 += 1

        elif word.lower() in gensim_model:
            word_vector = gensim_model.wv.word_vec(word.lower())
            syn0[idx, :] = word_vector
            match2 += 1

    match = match1 + match2
    matching_rate = 100 * (match / num_word_vocab)

    logger.info(f"Found \x1b[31m{matching_rate:.2f}%\x1b[0m words in pre-trained vocab")  # NOQA
    logger.info(f"- num_word_vocab: \x1b[31m{num_word_vocab}\x1b[0m")
    logger.info(f"- match1: \x1b[31m{match1}\x1b[0m, match2: \x1b[31m{match2}\x1b[0m")  # NOQA
    return syn0


@click.command()
@click.argument("config", type=str)
@click.option("--device", type=int, default=-1)
@click.option("--seed", type=int, default=31)
def run_training(config: str, device: int, seed: int):
    params = yaml.load(open(config, encoding="utf-8"))

    if device >= 0:
        chainer.cuda.get_device(device).use()

    set_seed(seed, device)

    configs = ConfigParser.parse(config)

    vocab = Vocabulary.prepare(configs)
    num_word_vocab = max(vocab.dictionaries["word2idx"].values()) + 1
    num_char_vocab = max(vocab.dictionaries["char2idx"].values()) + 1
    num_tag_vocab = max(vocab.dictionaries["tag2idx"].values()) + 1

    model = BiLSTM_CRF(configs, num_word_vocab, num_char_vocab, num_tag_vocab)

    transformer = DatasetTransformer(vocab)
    transform = transformer.transform

    external_configs = configs["external"]
    if "word_vector" in external_configs:
        syn0 = model.embed_word.W.data
        _, word_dim = syn0.shape
        pre_word_dim = vocab.gensim_model.vector_size
        if word_dim != pre_word_dim:
            msg = "Mismatch vector size between model and pre-trained word vectors"  # NOQA
            msg += f"(model: \x1b[31m{word_dim}\x1b[0m"
            msg += f", pre-trained word vector: \x1b[31m{pre_word_dim}\x1b[0m"
            raise Exception(msg)

        word2idx = vocab.dictionaries["word2idx"]
        syn0 = prepare_pretrained_word_vector(
            word2idx, vocab.gensim_model, syn0, num_word_vocab)
        model.set_pretrained_word_vectors(syn0)

    train_iterator = create_iterator(vocab, configs, "train", transform)
    valid_iterator = create_iterator(vocab, configs, "validation", transform)
    test_iterator = create_iterator(vocab, configs, "test", transform)

    if device >= 0:
        model.to_gpu(device)

    optimizer = create_optimizer(configs)
    optimizer.setup(model)
    optimizer = add_hooks(optimizer, configs)

    updater = T.StandardUpdater(
        train_iterator, optimizer, converter=converter, device=device
    )

    params = configs.export()
    params["num_word_vocab"] = num_word_vocab
    params["num_char_vocab"] = num_char_vocab
    params["num_tag_vocab"] = num_tag_vocab

    epoch = configs["iteration"]["epoch"]
    trigger = (epoch, "epoch")

    model_path = configs["output"]
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.isoformat()
    output_path = Path(f"{model_path}.{timestamp_str}")

    trainer = T.Trainer(updater, trigger, out=output_path)
    save_args(params, output_path)
    msg = f"Create \x1b[31m{output_path}\x1b[0m for saving model snapshots"
    logging.debug(msg)

    entries = ["epoch", "iteration", "elapsed_time", "lr", "main/loss"]
    entries += ["validation/main/loss", "validation/main/fscore"]
    entries += ["validation_1/main/loss", "validation_1/main/fscore"]

    valid_evaluator = NamedEntityEvaluator(
        valid_iterator, model, transformer.itransform, converter, device=device
    )

    test_evaluator = NamedEntityEvaluator(
        test_iterator, model, transformer.itransform, converter, device=device
    )

    epoch_trigger = (1, "epoch")
    snapshot_filename = "snapshot_epoch_{.updater.epoch:04d}"
    trainer.extend(valid_evaluator, trigger=epoch_trigger)
    trainer.extend(test_evaluator, trigger=epoch_trigger)
    trainer.extend(E.observe_lr(), trigger=epoch_trigger)
    trainer.extend(E.LogReport(trigger=epoch_trigger))
    trainer.extend(E.PrintReport(entries=entries), trigger=epoch_trigger)
    trainer.extend(E.ProgressBar(update_interval=20))
    trainer.extend(E.snapshot_object(
        model, filename=snapshot_filename), trigger=(1, "epoch"))

    if "learning_rate_decay" in params:
        logger.debug("Enable Learning Rate decay")
        trainer.extend(
            LearningRateDecay(
                "lr", params["learning_rate"], params["learning_rate_decay"]),
            trigger=epoch_trigger,
        )

    trainer.run()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    fmt = "%(asctime)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)

    run_training()
