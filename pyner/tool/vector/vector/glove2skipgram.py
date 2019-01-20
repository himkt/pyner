from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tmp_file = '/tmp/w2v.tmp'
    glove2word2vec(args.input_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    print('loaded GloVe embeddings')

    output_file = output_dir / input_file.stem
    model.save(output_file.as_posix())
    print('saved model')

    vocabulary = sorted(list(model.wv.vocab.keys()))
    vocabulary_path = output_dir / 'vocab.words.txt'
    with open(vocabulary_path, 'w') as file:
        for word in vocabulary:
            print(word, file=file)
