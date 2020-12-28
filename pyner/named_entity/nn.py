from itertools import accumulate, chain

import chainer
import chainer.functions as F
import chainer.links as L


class CharLSTM_Encoder(chainer.Chain):
    def __init__(
        self,
        n_char_vocab: int,
        n_layers: int,
        char_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        char_initializer=None,
    ):
        super(CharLSTM_Encoder, self).__init__()

        with self.init_scope():
            self.char_embed = L.EmbedID(
                in_size=n_char_vocab, out_size=char_dim, initialW=char_initializer
            )

            self.char_level_bilstm = L.NStepBiLSTM(
                n_layers=n_layers,
                in_size=char_dim,
                out_size=hidden_dim,
                dropout=dropout_rate,
            )

    def forward(self, char_sentences):
        flatten_char_sentences = list(chain.from_iterable(char_sentences))
        batch_size = len(flatten_char_sentences)

        offsets = list(accumulate(len(w) for w in flatten_char_sentences))
        char_embs_flatten = self.char_embed(
            self.xp.concatenate(flatten_char_sentences, axis=0)
        )  # NOQA
        char_embs = F.split_axis(char_embs_flatten, offsets[:-1], axis=0)

        hs, _, _ = self.char_level_bilstm(None, None, char_embs)
        char_features = hs.transpose((1, 0, 2))
        char_features = char_features.reshape(batch_size, -1)
        return char_features


class CharCNN_Encoder(chainer.Chain):
    def __init__(
        self,
        n_char_vocab: int,
        n_layers: int,
        char_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        char_initializer=None,
    ):
        super(CharCNN_Encoder, self).__init__()

        with self.init_scope():
            self.char_embed = L.EmbedID(
                in_size=n_char_vocab, out_size=char_dim, initialW=char_initializer
            )
            self.char_level_cnn = L.Convolution1D(
                in_channels=char_dim, out_channels=2 * hidden_dim, ksize=3
            )

    def forward(self, char_sentences):
        flatten_char_sentences = list(chain.from_iterable(char_sentences))
        char_sentence = F.pad_sequence(flatten_char_sentences, padding=1)
        char_embs = self.char_embed(char_sentence)
        cnn_output = self.char_level_cnn(char_embs.transpose((0, 2, 1)))
        return F.max(cnn_output, axis=2)


if __name__ == "__main__":
    import numpy

    cnn = CharCNN_Encoder(10, 1, 20, 100, 0.0)
    # cnn = CharLSTM_Encoder(10, 1, 4, 8, 0.0)
    a = [[1, 3, 0, 2], [1, 3, 0, 2], [3, 4], [4, 5, 1, 3, 0, 2], [3, 4]]
    a = [[numpy.asarray(_a, dtype=numpy.int32) for _a in a]]
    print(cnn.forward(a).shape)
