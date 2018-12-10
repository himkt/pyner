from pyner.initializer import XavierInitializer
from chainer import reporter

import chainer.functions as F
import chainer.links as L
import chainer
import logging
import gensim
import numpy


logger = logging.getLogger(__name__)


class BiLSTM_CRF(chainer.Chain):
    """
    BiLSTM-CRF: Bidirectional LSTM + Conditional Random Field as a decoder
    """
    # manage with dictionary
    def __init__(self, params, word2idx=None, label_matrix=None):
        super(BiLSTM_CRF, self).__init__()

        # word encoder
        self.n_word_vocab = params.get('n_word_vocab')
        self.word_dim = params.get('word_dim')
        self.word_hidden_dim = params.get('word_hidden_dim')
        self.initialW_embedf = params.get('word_vector')
        self.lower = params.get('lower')
        self.word2idx = word2idx

        # char encoder
        self.n_char_vocab = params.get('n_char_vocab')
        self.char_dim = params.get('char_dim')
        self.char_hidden_dim = params.get('char_hidden_dim')
        self.n_char_hidden_layer = 1
        self.classifier = params.get('word_classifier')

        # additional features (dictionary)
        self.n_label_vocab = params.get('n_label_vocab')
        self.dictionary = params.get('dictionary')
        self.label_matrix = label_matrix

        # additional features (classifier)
        self.classifier = params.get('classifier')
        self.classifier_output = params.get('classifier::output')
        self.classifier_usage = params.get('classifier::usage', 'lstm')
        self.metric = params.get('classifier::metric')

        # integrated word encoder
        self.word_hidden_dim = params.get('word_hidden_dim')
        self.n_tag_vocab = params.get('n_tag_vocab')
        self.n_word_hidden_layer = 1  # same as Lample

        # transformer
        self.linear_input_dim = 0

        # decoder
        self.n_tag_vocab = params.get('n_tag_vocab')

        # feature extractor (BiLSTM)
        self.internal_hidden_dim = 0
        self.dropout_rate = params.get('dropout', 0)

        # param initializer
        # approx: https://github.com/glample/tagger/blob/master/utils.py#L44
        # this is same as Xavier initialization
        # see also He initialization
        self.initializer = XavierInitializer()

        # init word vectors
        self._initialize_word_embeddings()

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

    def _initialize_word_embeddings(self):
        if self.initialW_embedf is None:
            shape = [self.n_word_vocab, self.word_dim]
            initialW_embed = self.xp.zeros(shape)
            self.initializer(initialW_embed)
            self.initialW_embed = initialW_embed
            logger.debug('Initialize embeddings randomly')
            return

        logger.debug(f'Initialize embeddings using {self.initialW_embedf}')
        gensim_model = gensim.models.KeyedVectors.load(self.initialW_embedf)
        word_dim = gensim_model.wv.vector_size

        n_word_vocab = len(self.word2idx)
        shape = [n_word_vocab, word_dim]

        # if lowercased word is in pre-trained embeddings,
        # increment match2
        match1, match2 = 0, 0

        initialW_embed = numpy.zeros(shape)
        self.initializer(initialW_embed)  # init

        for word, idx in self.word2idx.items():
            if word in gensim_model:
                word_vector = gensim_model.wv.word_vec(word)
                initialW_embed[idx, :] = word_vector
                match1 += 1

            elif self.lower and word.lower() in gensim_model:
                word_vector = gensim_model.wv.word_vec(word.lower())
                initialW_embed[idx, :] = word_vector
                match2 += 1

        match = match1 + match2
        matching_rate = 100 * (match/n_word_vocab)
        logger.info(f'Found {matching_rate:.2f}% words in pre-trained vocab')
        logger.info(f'- n_word_vocab: {n_word_vocab}')
        logger.info(f'- match1: {match1}, match2: {match2}')
        self.initialW_embed = initialW_embed

    def _setup_word_encoder(self):
        if self.word_dim is None:
            return

        logger.debug('Use word level encoder')
        self.embed_word = L.EmbedID(self.n_word_vocab,
                                    self.word_dim,
                                    initialW=self.initialW_embed)

    def _setup_char_encoder(self):
        if self.char_dim is None:
            return

        logger.debug('Use character level encoder')
        self.embed_char = L.EmbedID(self.n_char_vocab, self.char_dim)
        self.internal_hidden_dim += 2*self.char_hidden_dim

        self.char_level_bilstm = L.NStepBiLSTM(self.n_char_hidden_layer,
                                               self.char_dim,
                                               self.char_hidden_dim,
                                               self.dropout_rate)

    def _setup_feature_extractor(self):
        # ref: https://github.com/glample/tagger/blob/master/model.py#L256
        self.internal_hidden_dim += self.word_hidden_dim
        self.linear_input_dim += 2*self.word_hidden_dim

        self.word_level_bilstm = L.NStepBiLSTM(self.n_word_hidden_layer,
                                               self.internal_hidden_dim,
                                               self.word_hidden_dim,
                                               self.dropout_rate)
        self.linear = L.Linear(self.linear_input_dim, self.n_tag_vocab,
                               initialW=self.initializer)

    def _setup_decoder(self):
        self.crf = L.CRF1d(self.n_tag_vocab, initialW=self.initializer)

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

        if self.dictionary is not None:
            wdic = self.lookuper(word_sentence)
            word_features.append(wdic)

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
