from pathlib import Path
import logging
import re


SPECIAL_SYMBOLS = ["<UNK>", "<PAD>"]
FIELDS_ALL = ["word", "char", "tag"]
FIELDS_PREPROCESSED = ["word", "char"]
FIELDS_NEED_SPECIAL_SYMBOLS = ["word", "char"]


logger = logging.getLogger(__name__)


def _replace_zero(ws):
    ws = [re.sub(r"\d", "0", w) for w in ws]
    return ws


def _lowercase(ws):
    ws = [w.lower() for w in ws]
    return ws


def _insert_special_symbols(vocabulary):
    for symbol in SPECIAL_SYMBOLS:
        vocabulary[symbol] = len(vocabulary)
    return vocabulary


class Vocabulary:
    def __init__(self, configs):
        if "external" not in configs:
            raise Exception("External configurations are not given")

        if "preprocessing" not in configs:
            raise Exception("Preprocessing configurations are not found")

        external_configs = configs["external"]
        preprocessing_configs = configs["preprocessing"]

        self.__name__ = "Vocabulary"
        self.replace_zero = preprocessing_configs.get("replace_zero", False)
        self.lower = preprocessing_configs.get("lower", False)
        self.data_path = Path(external_configs.get("data_dir"))
        self.gensim_model_path = external_configs.get("word_vector")
        self.vocab_merge_operator = external_configs.get(
            "vocab_merge_operator", "union"
        )

        self.dictionaries = {}
        self.vocab_arr = {}

        logger.debug(f"Zero normalization: \x1b[31m{self.replace_zero}\x1b[0m")
        logger.debug(f"Lowercase: \x1b[31m{self.lower}\x1b[0m")

    def _process(self, vocab):
        if self.replace_zero:
            vocab = _replace_zero(vocab)

        if self.lower:
            vocab = _lowercase(vocab)

        return vocab

    def _compile(self):
        logger.debug("Compiling vocabularies...")
        for field, vocab in self.vocab_arr.items():
            if field in FIELDS_PREPROCESSED:
                vocab = self._process(vocab)
            vocab_arr = list(set(vocab))
            self.vocab_arr[field] = vocab_arr

        if self.gensim_model_path:
            self._load_pretrained_word_vectors(self.gensim_model_path)

        for name in self.vocab_arr.keys():
            # NOTE python dictionaries are unordered
            self.vocab_arr[name] = sorted(self.vocab_arr[name])
            vocabulary = {
                w: i for i, w in enumerate(sorted(self.vocab_arr[name]))
            }  # NOQA
            if name in FIELDS_NEED_SPECIAL_SYMBOLS:
                vocabulary = _insert_special_symbols(vocabulary)
            self.dictionaries[f"{name}2idx"] = vocabulary

    def _read(self, vocab_file):
        with open(vocab_file, encoding="utf-8") as vocab_file:
            vocab_txt = vocab_file.read()
            vocab_txt = vocab_txt.rstrip("\n")
            vocab_arr = vocab_txt.split("\n")
            return vocab_arr

    def _load_vocab(self, field):
        file_name = f"vocab.{field}s.txt"
        vocab_path = self.data_path / file_name

        if vocab_path.exists():
            vocab_arr = self._read(vocab_path.as_posix())
            self.vocab_arr[field] = vocab_arr

    @classmethod
    def prepare(cls, params, fields=FIELDS_ALL):
        vocab = cls(params)

        for field in fields:
            vocab._load_vocab(field)

        vocab._compile()
        return vocab

    def load_word_sentences(self, file):
        sentences = []
        with open(file, encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                words = line.split(" ")
                words = self._process(words)
                sentences.append(words)
        return sentences

    def load_tag_sentences(self, file):
        sentences = []
        with open(file, encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                sentences.append(line.split(" "))
        return sentences

    @staticmethod
    def _update_vocabulary(va, vb, operator):
        if operator == "intersection":
            vc = va & vb
        elif operator == "union":
            vc = va | vb
        else:
            return Exception("Unknown operator is specified")

        num_vocab = len(vc)
        msg = f" num_vocab: \x1b[31m{num_vocab}\x1b[0m"
        msg += f" (with set operator \x1b[31m{operator}\x1b[0m)"
        logger.debug(msg)
        return vc

    def _load_pretrained_word_vectors(self, word_vector_path):
        from gensim.models import KeyedVectors

        msg = f"Load word vectors from \x1b[31m{word_vector_path}\x1b[0m"
        logger.debug(msg)
        self.gensim_model = KeyedVectors.load(word_vector_path)
        vocab_arr = list(self.gensim_model.vocab.keys())
        vocab_arr = self._process(vocab_arr)
        gensim_model_vocab = list(vocab_arr)

        va = set(self.vocab_arr["word"])
        vb = set(gensim_model_vocab)
        vc = self._update_vocabulary(va, vb, self.vocab_merge_operator)
        self.vocab_arr["word"] = list(vc)
