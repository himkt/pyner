import logging
from pathlib import Path

import chainer.cuda
import chainer.dataset as D
import numpy as np

logger = logging.getLogger(__name__)


def update_instances(train_datas, params, mode):
    train_size = params.get("train_size", 1.0)
    if train_size <= 0 or 1 <= train_size:
        assert Exception("train_size must be in (0, 1]")

    rate = 100 * train_size
    n_instances = int(train_size * len(train_datas))
    logger.debug(f"Use {n_instances} example for {mode} ({rate:.2f}%)")

    result = [[] for _ in train_datas[0][0]]
    for instance in train_datas[:n_instances]:
        for index, sentence in enumerate(list(zip(*instance))):
            result[index].append(sentence)
    return result


def converter(batch, device=-1):
    xp = chainer.cuda.cupy if device >= 0 else np

    # transpose
    word_sentences, char_sentences, tag_sentences = list(zip(*batch))
    wss, css, tss = list(zip(*batch))

    # make ndarray
    wss = [xp.asarray(ws, dtype=xp.int32) for ws in wss]
    tss = [xp.asarray(ts, dtype=xp.int32) for ts in tss]
    css = [[xp.asarray(c, dtype=xp.int32) for c in cs] for cs in css]
    return (wss, css), tss


class DatasetTransformer:
    def __init__(self, vocab):
        self.word2idx = vocab.dictionaries["word2idx"]
        self.char2idx = vocab.dictionaries["char2idx"]
        self.tag2idx = vocab.dictionaries["tag2idx"]

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.replace_zero = vocab.replace_zero

    @staticmethod
    def _to_id(elems, dictionary):
        unk_id = dictionary.get("<UNK>")
        es = [dictionary.get(e, unk_id) for e in elems]
        return es

    def transform(self, word_sentence, tag_sentence):
        wordid_sentence = self._to_id(word_sentence, self.word2idx)
        tagid_sentence = self._to_id(tag_sentence, self.tag2idx)
        charid_sentence = [
            self._to_id(cs, self.char2idx) for cs in word_sentence
        ]  # NOQA
        return wordid_sentence, charid_sentence, tagid_sentence

    def itransform(self, wordid_sentences, tagid_sentences):
        """
        convert wordids and tagids into words and tags
        if you want to itransform iterator's output:
        > (words, _), tags = batch
        > itransform(words, tags)
        """
        sentences = zip(wordid_sentences, tagid_sentences)
        return [self._itransform(ws, ts) for ws, ts in sentences]

    def _itransform(self, wordid_sentence, tagid_sentence):
        wordid_sentence = chainer.cuda.to_cpu(wordid_sentence)
        tagid_sentence = chainer.cuda.to_cpu(tagid_sentence)
        word_sentence = [self.idx2word[wid] for wid in wordid_sentence]
        tag_sentence = [self.idx2tag[tid] for tid in tagid_sentence]

        return word_sentence, tag_sentence


class SequenceLabelingDataset(D.DatasetMixin):
    def __init__(self, vocab, params, mode, transform):
        data_dir = Path(params["data_dir"])
        data_path = data_dir / f"{mode}.txt"
        datas = vocab.load_word_sentences(data_path)
        word_sentences, tag_sentences = update_instances(datas, params, mode)
        self.word_sentences = word_sentences
        self.tag_sentences = tag_sentences

        self.num_sentences = len(word_sentences)
        self.transform = transform

    def __len__(self):
        return self.num_sentences

    def get_example(self, i):
        word_line = self.word_sentences[i]
        tag_line = self.tag_sentences[i]
        return self.transform(word_line, tag_line)
