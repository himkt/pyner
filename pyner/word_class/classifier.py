from chainer import reporter

import chainer.functions as F
import chainer.links as L
import chainer
import logging
import numpy


logger = logging.getLogger(__name__)


class CharacterBasedClassifier(chainer.Chain):

    def __init__(self, params):
        super(CharacterBasedClassifier, self).__init__()

        # character
        self.n_char_vocab = params.get('n_char_vocab')
        self.char_dim = params.get('char_dim')
        self.char_hidden_dim = params.get('char_hidden_dim')
        self.n_hidden_layer = params.get('n_hidden_layer')

        # decoder
        self.n_label_vocab = params.get('n_label_vocab')

        # fine-tuning
        self.fine_tune = params.get('fine_tune')

        if self.fine_tune is not None:
            logger.debug(f'Fix {self.fine_tune}')

        with self.init_scope():
            self._setup_char_encoder()
            self._setup_decoder()

    def _setup_char_encoder(self):
        self.char_embed = L.EmbedID(self.n_char_vocab, self.char_dim)
        self.char_level_bilstm = L.NStepBiLSTM(self.n_hidden_layer,
                                               self.char_dim,
                                               self.char_hidden_dim, 0.3)

    def _setup_decoder(self):
        self.output_layer = L.Linear(2*self.char_hidden_dim,
                                     self.n_label_vocab)

    def extract_hidden(self, batch):
        char_features = []

        for record in batch:
            char_emb = self.char_embed(
                self.xp.asarray(record.astype(numpy.int32)))
            char_features.append(char_emb)

            if self.fine_tune == 'embedding':
                char_emb.unchain_backward()

        batch_size = len(char_features)
        shape = [2*self.n_hidden_layer, batch_size, self.char_hidden_dim]
        h_0_data = self.xp.random.uniform(-1, 1, shape).astype(self.xp.float32)
        c_0_data = self.xp.random.uniform(-1, 1, shape).astype(self.xp.float32)
        h_0 = chainer.Variable(h_0_data)
        c_0 = chainer.Variable(c_0_data)

        _, _, hs = self.char_level_bilstm(h_0, c_0, char_features)
        h_vec = [h[-1] for h in hs]
        h_vec = F.vstack(h_vec)

        return h_vec

    def return_prob(self, batch):
        features = self.extract_hidden(batch)
        if self.fine_tune == 'lstm':
            features.unchain_backward()

        features = self.output_layer(features)
        if self.fine_tune == 'classifier':
            features.unchain_backward()

        output = F.softmax(features)
        return output

    def __call__(self, batch, labels):
        features = self.return_prob(batch)

        if labels is not None:
            loss = F.softmax_cross_entropy(features, labels)

        reporter.report({'loss': loss}, self)
        return loss

    def predict(self, batch):
        features = self.return_prob(batch)
        return self.xp.argmax(features.data, axis=1)


if __name__ == '__main__':
    params = {}
    params['n_char_vocab'] = 50
    params['char_dim'] = 20
    params['char_hidden_dim'] = 25
    params['n_label_vocab'] = 3

    model = CharacterBasedClassifier(params)

    inputs = []
    inputs.append(numpy.asarray([0, 10, 2, 4]))
    inputs.append(numpy.asarray([0, 10, 2, 4]))
    inputs.append(numpy.asarray([1, 4, 5]))

    outputs = numpy.asarray([0, 1, 0])
    print(model(inputs, outputs))
