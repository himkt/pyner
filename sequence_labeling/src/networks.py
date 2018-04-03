import chainer
import numpy


class CRF(chainer.Chain):
    """
    Simple Neural CRF

    This model does not have any window-based features: weak performance
    """

    def __init__(self, n_vocab, n_postag, n_label, word_dim, postag_dim, initialW):  # NOQA
        super(CRF, self).__init__()
        feature_dim = word_dim + postag_dim

        with self.init_scope():
            self.embed_word = chainer.links.EmbedID(n_vocab, word_dim)
            self.embed_postag = chainer.links.EmbedID(n_postag, postag_dim)
            self.linear = chainer.links.Linear(feature_dim, n_label)
            self.crf = chainer.links.CRF1d(n_label)

    def __call__(self, words, postags, labels):
        features = self.__extract__(words, postags)
        features = chainer.functions.transpose_sequence(features)
        labels = chainer.functions.transpose_sequence(labels)
        loss = self.crf(features, labels)
        chainer.reporter.report({'loss': loss.data}, self)
        return loss

    def __extract__(self, words, postags):
        features = []
        for word, postag in zip(words, postags):
            word_ = self.embed_word(word)
            postag_ = self.embed_postag(postag)
            feature = chainer.functions.concat([word_, postag_], axis=1)
            feature = self.linear(feature)
            features.append(feature)
        return features

    def predict(self, words, postags):
        ys = self.__extract__(words, postags)
        _, pathes = self.crf.argmax(ys)
        return pathes


class BiLSTM(chainer.Chain):
    """
    Vanilla BiLSTM: Bidirectional LSTM + softmax
    """

    def __init__(self, n_vocab, n_tag, n_label, word_dim, postag_dim, hidden_dim, initialW):  # NOQA
        super(BiLSTM, self).__init__()
        feature_dim = word_dim + postag_dim

        with self.init_scope():
            self.embed_word = chainer.links.EmbedID(n_vocab, word_dim, initialW=initialW)  # NOQA
            self.embed_postag = chainer.links.EmbedID(n_tag, postag_dim)
            self.bilstm = chainer.links.NStepBiLSTM(1, feature_dim, hidden_dim, 0.8)  # NOQA
            self.linear = chainer.links.Linear(2*hidden_dim, n_label)

    def __call__(self, words, postags, labels):
        ys = self.__extract__(words, postags)

        accum_loss = 0.0
        for y, label_seq in zip(ys, labels):
            accum_loss += chainer.functions.softmax_cross_entropy(y, label_seq)

        chainer.reporter.report({'loss': accum_loss.data}, self)
        return accum_loss

    def __extract__(self, words, postags):
        features = []
        for words_, postags_ in zip(words, postags):

            words_ = self.embed_word(words_)
            postags_ = self.embed_postag(postags_)
            feature = chainer.functions.concat([words_, postags_])
            features.append(feature)

        cy, hy, ys = self.bilstm(cx=None, hx=None, xs=features)
        ys = [self.linear(chainer.functions.tanh(y)) for y in ys]
        return ys

    def predict(self, words, postags):
        ys = self.__extract__(words, postags)
        ys = [chainer.functions.argmax(chainer.functions.softmax(y), axis=1).data for y in ys]  # NOQA
        return ys


class BiLSTM_CRF(chainer.Chain):
    """
    BiLSTM-CRF: Bidirectional LSTM + Conditional Random Field as a decoder
    """

    def __init__(self, n_vocab, n_postag, n_label, word_dim, postag_dim, hidden_dim, initialW):  # NOQA
        super(BiLSTM_CRF, self).__init__()
        feature_dim = word_dim + postag_dim

        with self.init_scope():
            self.embed_word = chainer.links.EmbedID(n_vocab, word_dim, initialW=initialW)  # NOQA
            self.embed_postag = chainer.links.EmbedID(n_postag, postag_dim)
            self.bilstm = chainer.links.NStepBiLSTM(1, feature_dim, hidden_dim, 0.8)  # NOQA
            self.linear = chainer.links.Linear(2*hidden_dim, n_label)
            self.crf = chainer.links.CRF1d(n_label)

    def __call__(self, words, postags, labels):
        features = self.__extract__(words, postags)
        features = chainer.functions.transpose_sequence(features)
        labels = chainer.functions.transpose_sequence(labels)

        loss = self.crf(features, labels)
        chainer.reporter.report({'loss': loss.data}, self)
        return loss

    def __extract__(self, words, postags):
        features = []
        for word, postag in zip(words, postags):
            word_ = self.embed_word(word)
            postag_ = self.embed_postag(postag)
            feature = chainer.functions.concat([word_, postag_], axis=1)
            features.append(feature)

        cy, hy, hs = self.bilstm(None, None, features)
        features = [self.linear(chainer.functions.tanh(h)) for h in hs]
        return features

    def predict(self, words, postags):
        features = self.__extract__(words, postags)
        _, pathes = self.crf.argmax(features)
        return pathes


class Semi_BiLSTM_CRF(chainer.Chain):
    """
    Semi_BiLSTM_CRF: BiLSTM CRF with Multitasking

    http://www.aclweb.org/anthology/P17-1194
    """

    def __init__(self, n_vocab, n_postag, n_lm_vocab, n_label,
                 word_dim, postag_dim, hidden_dim, fw_dim, bw_dim, initialW,
                 gamma=1.0, trainW=True):
        super(Semi_BiLSTM_CRF, self).__init__()
        self.gamma = gamma
        feature_dim = word_dim + postag_dim

        with self.init_scope():
            self.embed_word = chainer.links.EmbedID(n_vocab, word_dim, initialW=initialW)  # NOQA
            self.embed_postag = chainer.links.EmbedID(n_postag, postag_dim)
            self.forward_lstm = chainer.links.NStepLSTM(1, feature_dim, hidden_dim, 0.8)  # NOQA
            self.forward_lstm_linear = chainer.links.Linear(hidden_dim, fw_dim)  # NOQA
            self.forward_lm_linear = chainer.links.Linear(fw_dim, n_lm_vocab)  # NOQA
            self.backward_lstm = chainer.links.NStepLSTM(1, feature_dim, hidden_dim, 0.8)  # NOQA
            self.backward_lstm_linear = chainer.links.Linear(hidden_dim, bw_dim)  # NOQA
            self.backward_lm_linear = chainer.links.Linear(bw_dim, n_lm_vocab)  # NOQA
            self.linear = chainer.links.Linear(2*hidden_dim, n_label)
            self.crf = chainer.links.CRF1d(n_label)

    def __call__(self, words, postags, labels):
        bos = self.xp.asarray([0], dtype=self.xp.int32)
        eos = self.xp.asarray([1], dtype=self.xp.int32)
        forward_words = [chainer.functions.hstack([bos, words_[:-1]]) for words_ in words]  # NOQA
        backward_words = [chainer.functions.hstack([words_[1:], eos]) for words_ in words]  # NOQA
        label_linear, forward_lm_linear, backward_lm_linear = self.__extract__(words, postags)  # NOQA
        label_linear = chainer.functions.transpose_sequence(label_linear)
        labels = chainer.functions.transpose_sequence(labels)
        crfloss = self.crf(label_linear, labels)
        lm_loss = 0.0
        for forward_lm_linear_, forward_words_, backward_lm_linear_, backward_words_ in zip(forward_lm_linear, forward_words, backward_lm_linear, backward_words):  # NOQA
            forward_lm_loss = chainer.functions.softmax_cross_entropy(forward_lm_linear_, forward_words_)  # NOQA
            backward_lm_loss = chainer.functions.softmax_cross_entropy(backward_lm_linear_, backward_words_)  # NOQA
            lm_loss += (forward_lm_loss + backward_lm_loss)

        loss = (crfloss + self.gamma * lm_loss)
        chainer.reporter.report({'loss': loss.data}, self)
        return loss

    def __extract__(self, words, postags):
        features = []
        for words_, postags_ in zip(words, postags):
            words_ = self.embed_word(words_)
            postags_ = self.embed_postag(postags_)
            feature = chainer.functions.concat([words_, postags_])
            features.append(feature)

        *_, fys = self.forward_lstm(cx=None, hx=None, xs=features)
        *_, bys = self.backward_lstm(cx=None, hx=None, xs=features[::-1])
        bys = bys[::-1]

        label_linear = [self.linear(chainer.functions.tanh(chainer.functions.concat([fy, by]))) for fy, by in zip(fys, bys)]  # NOQA
        forward_lm_linear = [self.forward_lm_linear(self.forward_lstm_linear(chainer.functions.tanh(fy))) for fy in fys]  # NOQA
        backward_lm_linear = [self.backward_lm_linear(self.backward_lstm_linear(chainer.functions.tanh(by))) for by in bys]  # NOQA
        return label_linear, forward_lm_linear, backward_lm_linear

    def predict(self, words, postags):
        # FIXME 予測のときはlabel_linearだけで良い？
        label_linear, *_ = self.__extract__(words, postags)  # NOQA
        _, pathes = self.crf.argmax(label_linear)
        return pathes


class Char_BiLSTM_CRF(chainer.Chain):
    """
    Char_BiLSTM_CRF: BiLSTM CRF using both char and word level features

    Neural Architectures for Named Entity Recognition
    http://www.aclweb.org/anthology/N16-1030
    """

    def __init__(self, n_vocab, n_char, n_postag, n_label, word_dim, char_dim,
                 postag_dim, hidden_dim, char_hidden_dim, initialW):
        super(Char_BiLSTM_CRF, self).__init__()
        feature_dim = word_dim + 2*char_hidden_dim + postag_dim

        with self.init_scope():
            self.embed_word = chainer.links.EmbedID(n_vocab, word_dim, initialW)  # NOQA
            self.embed_char = chainer.links.EmbedID(n_vocab, char_dim)
            self.embed_postag = chainer.links.EmbedID(n_postag, postag_dim)
            self.feature_bilstm = chainer.links.NStepBiLSTM(1, feature_dim, hidden_dim, 0.8)  # NOQA
            self.char_bilstm = chainer.links.NStepBiLSTM(1, char_dim, char_hidden_dim, 0.8)  # NOQA
            self.linear = chainer.links.Linear(2*hidden_dim, n_label)
            self.crf = chainer.links.CRF1d(n_label)

    def __call__(self, words, chars, postags, labels):  # NOQA
        features = self.__extract__(words, chars, postags)  # NOQA
        features = chainer.functions.transpose_sequence(features)
        labels = chainer.functions.transpose_sequence(labels)

        loss = self.crf(features, labels)
        chainer.reporter.report({'loss': loss.data}, self)
        return loss

    def __extract__(self, words, chars, postags):  # NOQA
        features = []

        for word_seq, chars_seq, postag_seq in zip(words, chars, postags):  # NOQA
            word_repr = self.embed_word(word_seq)
            postag_repr = self.embed_postag(postag_seq)

            # TODO documentation
            _, _, chars_repr = self.char_bilstm(None, None, [self.embed_char(chars) for chars in chars_seq])  # NOQA
            chars_repr = chainer.functions.vstack([chainer.functions.tanh(chars_repr_[-1]) for chars_repr_ in chars_repr])  # NOQA

            feature = chainer.functions.concat([word_repr, chars_repr, postag_repr], axis=1)  # NOQA
            features.append(feature)

        cy, hy, hs = self.feature_bilstm(None, None, features)
        features = [self.linear(chainer.functions.tanh(h)) for h in hs]
        return features

    def predict(self, words, chars, postags):
        features = self.__extract__(words, chars, postags)
        _, pathes = self.crf.argmax(features)
        return pathes


if __name__ == '__main__':
    n_vocab, n_char, n_postag, n_label = 30, 5, 4, 5
    word_dim, char_dim, postag_dim, hidden_dim, char_hidden_dim = 20, 20, 10, 30, 10  # NOQA
    fw_dim, bw_dim = 20, 20

    words = [[1, 2, 0], [2, 3]]
    postags = [[1, 2, 3], [1, 1]]
    labels = [[1, 2, 1], [0, 1]]
    chars = [[[0, 1], [0], [1, 2]], [[0, 2], [3, 4]]]

    words = [numpy.asarray(seq, dtype=numpy.int32) for seq in words]  # NOQA
    chars = [numpy.asarray([numpy.asarray(char, dtype=numpy.int32) for char in chars]) for chars in chars]  # NOQA
    postags = [numpy.asarray(seq, dtype=numpy.int32) for seq in postags]  # NOQA
    labels = [numpy.asarray(seq, dtype=numpy.int32) for seq in labels]  # NOQA

    model = CRF(n_vocab, n_postag, n_label, word_dim, postag_dim)
    print(model(words, postags, labels))
    print(model.predict(words, postags))

    model = BiLSTM(n_vocab, n_postag, n_label, word_dim, postag_dim, hidden_dim)  # NOQA
    print(model(words, postags, labels))
    print(model.predict(words, postags))

    model = BiLSTM_CRF(n_vocab, n_postag, n_label, word_dim, postag_dim, hidden_dim)  # NOQA
    print(model(words, postags, labels))
    print(model.predict(words, postags))

    n_lm_vocab = 10
    model = Semi_BiLSTM_CRF(n_vocab, n_postag, n_lm_vocab, n_label, word_dim, postag_dim, hidden_dim, fw_dim, bw_dim)  # NOQA
    print(model(words, postags, labels))
    print(model.predict(words, postags))

    model = Char_BiLSTM_CRF(n_vocab, n_char, n_postag, n_label, word_dim, char_dim, postag_dim, hidden_dim, char_hidden_dim)  # NOQA
    print(model(words, chars, postags, labels))  # NOQA
    print(model.predict(words, chars, postags))
