from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    tmp_file = '/tmp/w2v.tmp'
    glove2word2vec(args.input_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    print('loaded GloVe embeddings')

    model.save(output_file.as_posix())
    print('saved model')
