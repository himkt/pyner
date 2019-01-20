from gensim.models import KeyedVectors
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path('./data/processed', input_file.name)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = KeyedVectors.load_word2vec_format(args.input_file)
    print('loaded pre-trained word2vec model')

    output_file = output_dir / 'model'
    model.save(output_file.as_posix())
    print('saved model')

    vocabulary = sorted(list(model.wv.vocab.keys()))
    vocabulary_path = output_dir / 'vocab.words.txt'
    with open(vocabulary_path, 'w') as file:
        for word in vocabulary:
            print(word, file=file)
