import logging
from itertools import accumulate

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers, reporter

from pyner.named_entity.nn import CharLSTM_Encoder

logger = logging.getLogger(__name__)


class BiLSTM_CRF(chainer.Chain):
    """
    BiLSTM-CRF: Bidirectional LSTM + Conditional Random Field as a decoder
    """

    def __init__(
            self,
            configs,
            num_word_vocab,
            num_char_vocab,
            num_tag_vocab
    ):

        super(BiLSTM_CRF, self).__init__()
        if "model" not in configs:
            raise Exception("Model configurations are not found")

        model_configs = configs["model"]

        model_configs["num_word_vocab"] = num_word_vocab
        model_configs["num_char_vocab"] = num_char_vocab
        model_configs["num_tag_vocab"] = num_tag_vocab

        # word encoder
        self.num_word_vocab = model_configs.get("num_word_vocab")
        self.word_dim = model_configs.get("word_dim")
        self.word_hidden_dim = model_configs.get("word_hidden_dim")

        # char encoder
        self.num_char_vocab = model_configs.get("num_char_vocab")
        self.num_char_hidden_layers = 1
        self.char_dim = model_configs.get("char_dim")
        self.char_hidden_dim = model_configs.get("char_hidden_dim")

        # integrated word encoder
        self.num_word_hidden_layers = 1  # same as Lample
        self.word_hidden_dim = model_configs.get("word_hidden_dim")

        # transformer
        self.linear_input_dim = 0

        # decoder
        self.num_tag_vocab = model_configs.get("num_tag_vocab")

        # feature extractor (BiLSTM)
        self.internal_hidden_dim = 0
        self.dropout_rate = model_configs.get("dropout", 0)

        # param initializer
        # approx: https://github.com/glample/tagger/blob/master/utils.py#L44
        self.initializer = initializers.GlorotUniform()

        # setup links with given params
        with self.init_scope():
            self._setup_word_encoder()
            self._setup_char_encoder()
            self._setup_feature_extractor()
            self._setup_decoder()

        logger.debug(f"Dropout rate: \x1b[31m{self.dropout_rate}\x1b[0m")
        logger.debug(f"Word embedding dim: \x1b[31m{self.word_dim}\x1b[0m")
        logger.debug(f"Char embedding dim: \x1b[31m{self.char_dim}\x1b[0m")

    def set_pretrained_word_vectors(self, syn0):
        self.embed_word.W.data = syn0

    def _setup_word_encoder(self):
        if self.word_dim is None:
            return

        logger.debug("Use word level encoder")
        self.embed_word = L.EmbedID(
            self.num_word_vocab,
            self.word_dim,
            initialW=self.initializer
        )

    def _setup_char_encoder(self):
        if self.char_dim is None:
            return

        logger.debug("Use character level encoder")
        self.char_level_encoder = CharLSTM_Encoder(
            self.num_char_vocab,
            self.num_char_hidden_layers,
            self.char_dim,
            self.char_hidden_dim,
            self.dropout_rate,
            char_initializer=self.initializer,
        )
        self.internal_hidden_dim += 2 * self.char_hidden_dim

    def _setup_feature_extractor(self):
        # ref: https://github.com/glample/tagger/blob/master/model.py#L256
        self.internal_hidden_dim += self.word_hidden_dim
        self.linear_input_dim += 2 * self.word_hidden_dim

        self.word_level_bilstm = L.NStepBiLSTM(
            n_layers=self.num_word_hidden_layers,
            in_size=self.internal_hidden_dim,
            out_size=self.word_hidden_dim,
            dropout=self.dropout_rate)

        self.linear = L.Linear(
            in_size=self.linear_input_dim,
            out_size=self.num_tag_vocab,
            initialW=self.initializer)

    def _setup_decoder(self):
        self.crf = L.CRF1d(
            n_label=self.num_tag_vocab,
            initial_cost=self.initializer)

    def forward(self, inputs, outputs, **kwargs):
        features = self.__extract__(inputs, **kwargs)
        loss = self.crf(features, outputs, transpose=True)

        _, pathes = self.crf.argmax(features, transpose=True)
        reporter.report({"loss": loss}, self)
        return loss

    def predict(self, batch, **kwargs):
        features = self.__extract__(batch)
        _, pathes = self.crf.argmax(features, transpose=True)
        return pathes

    def __extract__(self, batch, **kwargs):
        """
        :param batch: list of list, inputs
        inputs: (word_sentences, char_sentences)
        """
        word_sentences, char_sentences = batch
        offsets = list(accumulate(len(s) for s in word_sentences))

        lstm_inputs = []
        if self.word_dim is not None:
            word_repr = self.embed_word(self.xp.concatenate(word_sentences, axis=0))  # NOQA
            word_repr = F.dropout(word_repr, self.dropout_rate)
            lstm_inputs.append(word_repr)

        if self.char_dim is not None:
            # NOTE [[list[int]]
            char_repr = self.char_level_encoder(char_sentences)
            char_repr = F.dropout(char_repr, self.dropout_rate)
            lstm_inputs.append(char_repr)

        lstm_inputs = F.split_axis(F.concat(lstm_inputs, axis=1), offsets[:-1], axis=0)  # NOQA
        _, _, hs = self.word_level_bilstm(None, None, lstm_inputs)
        features = [self.linear(h) for h in hs]
        return features
