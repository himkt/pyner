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
                 model_config,
                 num_word_vocab,
                 num_char_vocab,
                 num_tag_vocab
                 ):

        super(BiLSTM_CRF, self).__init__()

        params = model_config
        params['num_word_vocab'] = num_word_vocab
        params['num_char_vocab'] = num_char_vocab
        params['num_tag_vocab'] = num_tag_vocab

        # word encoder
        self.num_word_vocab = params.get('num_word_vocab')
        self.word_dim = params.get('word_dim')
        self.word_hidden_dim = params.get('word_hidden_dim')

        # char encoder
        self.num_char_vocab = params.get('num_char_vocab')
        self.num_char_hidden_layers = 1
        self.char_dim = params.get('char_dim')
        self.char_hidden_dim = params.get('char_hidden_dim')

        # integrated word encoder
        self.num_word_hidden_layers = 1  # same as Lample
        self.word_hidden_dim = params.get('word_hidden_dim')

        # transformer
        self.linear_input_dim = 0

        # decoder
        self.num_tag_vocab = params.get('num_tag_vocab')

        # feature extractor (BiLSTM)
        self.internal_hidden_dim = 0
        self.dropout_rate = params.get('dropout', 0)

        # param initializer
        # approx: https://github.com/glample/tagger/blob/master/utils.py#L44
        self.initializer = initializers.GlorotUniform()

        # setup links with given params
        with self.init_scope():
            self._setup_word_encoder()
            self._setup_char_encoder()
            self._setup_feature_extractor()
            self._setup_decoder()

        logger.debug(f'Dropout rate: {self.dropout_rate}')
        logger.debug(f'Dim of word embeddings: {self.word_dim}')
        logger.debug(f'Dim of character embeddings: {self.char_dim}')

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
        word_features = []

        if self.word_dim is not None:
            wemb = self.embed_word(word_sentence)
            word_features.append(wemb)

        return F.hstack(word_features)

    def char_encode(self, char_inputs, **kwargs):
        if self.char_dim is None:
            return

        char_features = []
        char_embs = []
        for char_input in char_inputs:
            # TODO (himkt) remove this hacky workaround
            # if asarray is not provided,
            # sometimes char_input.shape be 0-d array
            char_input = self.xp.asarray(char_input, dtype=self.xp.int32)
            char_emb = self.embed_char(char_input)
            char_embs.append(char_emb)

        batch_size = len(char_embs)
        shape = [2, batch_size, self.char_hidden_dim]

        h_0, c_0 = self.create_init_state(shape)
        hs, _, _ = self.char_level_bilstm(h_0, c_0, char_embs)
        _, batch_size, _ = hs.shape
        hs = hs.transpose([1, 0, 2])
        hs = hs.reshape(batch_size, -1)
        char_features.append(hs)

        # final timestep for each sequence
        return F.hstack(char_features)

    def __extract__(self, batch, **kwargs):
        """
        :param batch: list of list, inputs
        inputs: (word_sentences, char_sentences)
        """
        lstm_inputs = []

        for word_sentence, char_sentence in zip(*batch):
            lstm_input = []

            word_repr = self.word_encode(word_sentence)
            if word_repr is not None:
                lstm_input.append(word_repr)

            char_repr = self.char_encode(char_sentence)
            if char_repr is not None:
                lstm_input.append(char_repr)

            lstm_input = F.concat(lstm_input, axis=1)
            lstm_input = F.dropout(lstm_input, self.dropout_rate)
            lstm_inputs.append(lstm_input)

        batch_size = len(batch[0])
        shape = [2, batch_size, self.word_hidden_dim]

        h_0, c_0 = self.create_init_state(shape)
        _, _, hs = self.word_level_bilstm(h_0, c_0, lstm_inputs)
        features = [self.linear(h) for h in hs]
        return features
