import logging
import pathlib

import click

from pyner.tool.corpus.common import (CorpusParser, enum, write_sentences,
                                      write_vocab)

SEED = 42
BOS = 0  # begin of step
EOS = 1  # end  of step
XXX = 2  # other


@click.command()
@click.option("--data-dir", default="./data/external/CoNLL2003", type=str)
@click.option("--output-dir", default="./data/processed/CoNLL2003", type=str)
@click.option("--format", default="iob2bio", type=str)
def main(data_dir: str, output_dir: str, format: str):
    logging.info("create dataset for CoNLL2003")

    data_path = pathlib.Path(data_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logging.info("create corpus parser")
    corpus_parser = CorpusParser(format)

    logging.info("parsing corpus for training")
    train_word_sentences, train_tag_sentences = corpus_parser.parse_file(
        data_path / "eng.iob.train", word_idx=0
    )
    train_words, train_chars, train_tags = enum(
        train_word_sentences, train_tag_sentences
    )

    logging.info("parsing corpus for validating")
    validation_word_sentences, validation_tag_sentences = corpus_parser.parse_file(  # NOQA
        data_path / "eng.iob.testa", word_idx=0
    )
    validation_words, validation_chars, validation_tags = enum(
        validation_word_sentences, validation_tag_sentences
    )

    logging.info("parsing corpus for testing")
    test_word_sentences, test_tag_sentences = corpus_parser.parse_file(
        data_path / "eng.iob.testb", word_idx=0
    )
    test_words, test_chars, test_tags = enum(test_word_sentences, test_tag_sentences)

    # NOTE create vocabularies only using training dataset
    words = train_words
    chars = train_chars
    tags = train_tags

    logging.info("Create training dataset")
    write_sentences("train", "words", train_word_sentences, output_path)
    write_sentences("train", "tags", train_tag_sentences, output_path)

    logging.info("Create validating dataset")
    write_sentences(
        "validation", "words", validation_word_sentences, output_path
    )  # NOQA
    write_sentences("validation", "tags", validation_tag_sentences, output_path)  # NOQA

    logging.info("Create testing dataset")
    write_sentences("test", "words", test_word_sentences, output_path)
    write_sentences("test", "tags", test_tag_sentences, output_path)

    logging.info("Create vocabulary")
    write_vocab("words", words, output_path)
    write_vocab("chars", chars, output_path)
    write_vocab("tags", tags, output_path)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    main()
