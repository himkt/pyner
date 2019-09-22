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
        char_initializer=None
    ):
        super(CharLSTM_Encoder, self).__init__()

        with self.init_scope():
            self.char_embed = L.EmbedID(
                in_size=n_char_vocab,
                out_size=char_dim,
                initialW=char_initializer)

            self.char_level_bilstm = L.NStepBiLSTM(
                n_layers=n_layers,
                in_size=char_dim,
                out_size=hidden_dim,
                dropout=dropout_rate)

    def forward(self, char_sentences):
        flatten_char_sentences = list(chain.from_iterable(char_sentences))
        batch_size = len(flatten_char_sentences)

        offsets = list(accumulate(len(w) for w in flatten_char_sentences))
        char_embs_flatten = self.char_embed(self.xp.concatenate(flatten_char_sentences, axis=0))  # NOQA
        char_embs = F.split_axis(char_embs_flatten, offsets[:-1], axis=0)

        hs, _, _ = self.char_level_bilstm(None, None, char_embs)
        char_features = hs.transpose((1, 0, 2))
        char_features = char_features.reshape(batch_size, -1)
        return char_features
