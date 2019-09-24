import logging
import pathlib

import click
from pyner.tool.corpus.common import CorpusParser
from pyner.tool.corpus.common import enum
from pyner.tool.corpus.common import write_sentences
from pyner.tool.corpus.common import write_vocab

SEED = 42
BOS  = 0  # begin of step
EOS  = 1  # end  of step
XXX  = 2  # other


@click.command()
@click.option("--data-dir", default="./data/external/CoNLL2003", type=str)
@click.option("--output-dir", default="./data/processed/CoNLL2003", type=str)
@click.option("--convert-rule", default="iob2bio", type=str)
@click.option("--delimiter", default=r" +", type=str)
def main(data_dir: str, output_dir: str, convert_rule: str, delimiter: str):
    logging.info("create dataset for CoNLL2003")

    data_path = pathlib.Path(data_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logging.info("create corpus parser")
    corpus_parser = CorpusParser(convert_rule, delimiter)

    logging.info("parsing corpus for training")
    train_word_sentences, train_tag_sentences = corpus_parser.parse_file(
        data_path / "eng.iob.train", word_idx=0
    )
    train_words, train_chars, train_tags = enum(
        train_word_sentences, train_tag_sentences
    )

    logging.info("parsing corpus for validating")
    valid_word_sentences, valid_tag_sentences = corpus_parser.parse_file(  # NOQA
        data_path / "eng.iob.testa", word_idx=0
    )
    _, _, valid_tags = enum(
        valid_word_sentences, valid_tag_sentences
    )

    logging.info("parsing corpus for testing")
    test_word_sentences, test_tag_sentences = corpus_parser.parse_file(
        data_path / "eng.iob.testb", word_idx=0
    )
    _, _, test_tags = enum(
        test_word_sentences, test_tag_sentences
    )

    for mode in ["train", "valid", "test"]:
        if mode == "train":
            sentences = list(zip(
                train_word_sentences,
                train_tag_sentences,
            ))
        elif mode == "valid":
            sentences = list(zip(
                valid_word_sentences,
                valid_tag_sentences,
            ))
        elif mode == "test":
            sentences = list(zip(
                test_word_sentences,
                test_tag_sentences,
            ))

        logging.info("Create %s dataset", mode)
        write_sentences(mode, sentences, output_path)

    # NOTE create vocabularies only using training dataset
    logging.info("Create vocabulary")
    vocab, char_vocab = train_words, train_chars
    tag_vocab = train_tags + valid_tags + test_tags
    write_vocab("words", vocab, output_path)
    write_vocab("chars", char_vocab, output_path)
    write_vocab("tags", tag_vocab, output_path)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    main()
