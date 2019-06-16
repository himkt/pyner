from itertools import accumulate, chain

from chainer import initializers
from chainer import reporter

import chainer.functions as F
import chainer.links as L
import chainer
import logging


logger = logging.getLogger(__name__)


class BiLSTM_CRF(chainer.Chain):
    """
    BiLSTM-CRF: Bidirectional LSTM + Conditional Random Field as a decoder
    """

    def __init__(self,
                 configs,
                 num_word_vocab,
                 num_char_vocab,
                 num_tag_vocab
                 ):

        super(BiLSTM_CRF, self).__init__()
        if 'model' not in configs:
            raise Exception('Model configurations are not found')

        model_configs = configs['model']

        model_configs['num_word_vocab'] = num_word_vocab
        model_configs['num_char_vocab'] = num_char_vocab
        model_configs['num_tag_vocab'] = num_tag_vocab

        # word encoder
        self.num_word_vocab = model_configs.get('num_word_vocab')
        self.word_dim = model_configs.get('word_dim')
        self.word_hidden_dim = model_configs.get('word_hidden_dim')

        # char encoder
        self.num_char_vocab = model_configs.get('num_char_vocab')
        self.num_char_hidden_layers = 1
        self.char_dim = model_configs.get('char_dim')
        self.char_hidden_dim = model_configs.get('char_hidden_dim')

        # integrated word encoder
        self.num_word_hidden_layers = 1  # same as Lample
        self.word_hidden_dim = model_configs.get('word_hidden_dim')

        # transformer
        self.linear_input_dim = 0

        # decoder
        self.num_tag_vocab = model_configs.get('num_tag_vocab')

        # feature extractor (BiLSTM)
        self.internal_hidden_dim = 0
        self.dropout_rate = model_configs.get('dropout', 0)

        # param initializer
        # approx: https://github.com/glample/tagger/blob/master/utils.py#L44
        self.initializer = initializers.GlorotUniform()

        # setup links with given params
        with self.init_scope():
            self._setup_word_encoder()
            self._setup_char_encoder()
            self._setup_feature_extractor()
            self._setup_decoder()

        logger.debug(f'Dropout rate: \x1b[31m{self.dropout_rate}\x1b[0m')  # NOQA
        logger.debug(f'Dim of word embeddings: \x1b[31m{self.word_dim}\x1b[0m')  # NOQA
        logger.debug(f'Dim of character embeddings: \x1b[31m{self.char_dim}\x1b[0m')  # NOQA

    def create_init_state(self, shape):
        h_0_data = self.xp.zeros(shape)
        self.initializer(h_0_data)
        h_0_data = h_0_data.astype(self.xp.float32)
        c_0_data = self.xp.zeros(shape)
        self.initializer(c_0_data)
        c_0_data = c_0_data.astype(self.xp.float32)
        h_0 = chainer.Variable(h_0_data)
        c_0 = chainer.Variable(c_0_data)
        return h_0, c_0

    def set_pretrained_word_vectors(self, syn0):
        self.embed_word.W.data = syn0

    def _setup_word_encoder(self):
        if self.word_dim is None:
            return

        logger.debug('Use word level encoder')
        self.embed_word = L.EmbedID(
            self.num_word_vocab,
            self.word_dim,
            initialW=self.initializer
        )

    def _setup_char_encoder(self):
        if self.char_dim is None:
            return

        logger.debug('Use character level encoder')
        self.embed_char = L.EmbedID(
            self.num_char_vocab,
            self.char_dim,
            initialW=self.initializer
        )

        self.internal_hidden_dim += 2*self.char_hidden_dim

        self.char_level_bilstm = L.NStepBiLSTM(
            self.num_char_hidden_layers,
            self.char_dim,
            self.char_hidden_dim,
            self.dropout_rate
        )

    def _setup_feature_extractor(self):
        # ref: https://github.com/glample/tagger/blob/master/model.py#L256
        self.internal_hidden_dim += self.word_hidden_dim
        self.linear_input_dim += 2*self.word_hidden_dim

        self.word_level_bilstm = L.NStepBiLSTM(
            self.num_word_hidden_layers,
            self.internal_hidden_dim,
            self.word_hidden_dim,
            self.dropout_rate
        )

        self.linear = L.Linear(
            self.linear_input_dim,
            self.num_tag_vocab,
            initialW=self.initializer
        )

    def _setup_decoder(self):
        self.crf = L.CRF1d(
            self.num_tag_vocab,
            initial_cost=self.initializer
        )

    def __call__(self, inputs, outputs, **kwargs):
        features = self.__extract__(inputs, **kwargs)
        loss = self.crf(features, outputs, transpose=True)

        _, pathes = self.crf.argmax(features, transpose=True)
        reporter.report({'loss': loss}, self)
        return loss

    def predict(self, batch, **kwargs):
        features = self.__extract__(batch)
        _, pathes = self.crf.argmax(features, transpose=True)
        return pathes

    def word_encode(self, word_sentence):
        word_features = self.embed_word(word_sentence)
        return word_features

    def char_encode(self, char_inputs, **kwargs):
        batch_size = len(char_inputs)
        offsets = list(accumulate(len(w) for w in char_inputs))
        char_embs_flatten = self.embed_char(
            self.xp.concatenate(char_inputs, axis=0))
        char_embs = F.split_axis(char_embs_flatten, offsets[:-1], axis=0)

        h_0, c_0 = self.create_init_state(
            (2, batch_size, self.char_hidden_dim))
        hy, _, hs = self.char_level_bilstm(h_0, c_0, char_embs)

        if batch_size == 1 and len(char_inputs[0]) == 1:
            # NOTE: See issue https://github.com/himkt/pyner/issues/38
            char_features = hs[0]
        else:
            char_features = hy.transpose((1, 0, 2)).reshape(batch_size, -1)
        return char_features

    def __extract__(self, batch, **kwargs):
        """
        :param batch: list of list, inputs
        inputs: (word_sentences, char_sentences)
        """
        word_sentences, char_sentences = batch
        offsets = list(accumulate(len(s) for s in word_sentences))

        lstm_inputs = []
        if self.word_dim is not None:
            word_repr = self.word_encode(
                self.xp.concatenate(word_sentences, axis=0))
            lstm_inputs.append(word_repr)
        if self.char_dim is not None:
            char_repr = self.char_encode(
                list(chain.from_iterable(char_sentences)))
            lstm_inputs.append(char_repr)
        lstm_inputs = F.split_axis(
            F.concat(lstm_inputs, axis=1), offsets[:-1], axis=0)

        h_0, c_0 = self.create_init_state(
            (2, len(word_sentences), self.word_hidden_dim))
        hs = self.word_level_bilstm(h_0, c_0, lstm_inputs)[-1]
        features = [self.linear(h) for h in hs]
        return features
