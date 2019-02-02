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

    model = KeyedVectors.load_word2vec_format(args.input_file)
    print('loaded pre-trained word2vec model')

    model.save(output_file.as_posix())
    print('saved model')
