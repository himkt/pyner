import logging
import pathlib

import click
from pyner.tool.corpus.common import (CorpusParser, enum, write_sentences,
                                      write_vocab)

SEED = 42
BOS  = 0  # begin of step
EOS  = 1  # end  of step
XXX  = 2  # other


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
    valid_word_sentences, valid_tag_sentences = corpus_parser.parse_file(  # NOQA
        data_path / "eng.iob.testa", word_idx=0
    )
    valid_words, valid_chars, valid_tags = enum(
        valid_word_sentences, valid_tag_sentences
    )

    logging.info("parsing corpus for testing")
    test_word_sentences, test_tag_sentences = corpus_parser.parse_file(
        data_path / "eng.iob.testb", word_idx=0
    )
    test_words, test_chars, test_tags = enum(
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

        logging.info(f"Create {mode} dataset")
        write_sentences(mode, sentences, output_path)

    # NOTE create vocabularies only using training dataset
    logging.info("Create vocabulary")
    vocab, char_vocab, tag_vocab = train_words, train_chars, train_tags
    write_vocab("words", vocab, output_path)
    write_vocab("chars", char_vocab, output_path)
    write_vocab("tags", tag_vocab, output_path)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    main()
