import pathlib

import click

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


@click.command()
@click.argument("input_file", type=str)  # NOQA
@click.argument("output_file", type=str)
@click.option("--format", default="word2vec", type=str)
def main(input_file: str, output_file: str, format: str):
    input_file = pathlib.Path(input_file)  # NOQA
    output_file = pathlib.Path(output_file)

    if format == "glove":
        tmp_file = "/tmp/w2v.tmp"
        glove2word2vec(input_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
        print("loaded GloVe embeddings")

    elif format == "word2vec":
        model = KeyedVectors.load_word2vec_format(input_file)
        print("loaded Word2vec embeddings")

    model.save(output_file.as_posix())
    print("saved model")


if __name__ == "__main__":
    main()
