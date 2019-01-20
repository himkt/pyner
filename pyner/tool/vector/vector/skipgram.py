from gensim.models import word2vec
from pathlib import Path
import argparse
import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')

    # https://github.com/tmikolov/word2vec/blob/master/word2vec.c
    parser.add_argument('--min-freq', default=5, type=int)
    parser.add_argument('--negative', default=5, type=int)
    parser.add_argument('--window', default=8, type=int)
    parser.add_argument('--dimension', default=100, type=int)
    parser.add_argument('--worker', default=32, type=int)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    skipgram_name = Path(args.input_file).stem
    skipgram_name += f'+dimension_{args.dimension:03d}'
    skipgram_name += f'+negative_{args.dimension:03d}'
    skipgram_name += f'+window_{args.window:03d}'
    logging.info(f'Word Embedding will be exported to {skipgram_name}')

    sentences = word2vec.LineSentence(args.input_file)
    model = word2vec.Word2Vec(sentences,
                              sg=1,
                              size=args.dimension,
                              window=args.window,
                              workers=args.worker,
                              min_count=args.min_freq,
                              hs=0,
                              seed=42,
                              negative=args.negative)

    output_path = Path('./data/processed', skipgram_name)
    output_path.mkdir(parents=True, exist_ok=True)

    skipgram_file = (output_path / 'model').as_posix()
    model.save(skipgram_file)

    vocab_arr = sorted(list(model.wv.vocab.keys()))
    vocab_path = output_path / 'vocab.words.txt'
    with open(vocab_path, 'w') as file:
        for word in vocab_arr:
            print(word, file=file)
