from chainer import reporter
import chainer.functions as F
import chainer.links as L

import chainer
import numpy
import six


class CRFTagger(chainer.Chain):

    def __init__(self, n_vocab, n_pos):
        super(CRFTagger, self).__init__()
        with self.init_scope():
            self.feature = L.EmbedID(n_vocab, n_pos)
            self.crf = L.CRF1d(n_pos)

    def __call__(self, xs, ys):
        # Before making a transpose, you need to sort two lists in descending
        # order of length.
        inds = numpy.argsort([-len(x) for x in xs]).astype('i')
        xs = [xs[i] for i in inds]
        ys = [ys[i] for i in inds]

        # Make transposed sequences.
        # Now xs[t] is a batch of words at time t.
        xs = F.transpose_sequence(xs)
        ys = F.transpose_sequence(ys)

        # h[i] is feature vector for each batch of words.
        hs = [self.feature(x) for x in xs]
        loss = self.crf(hs, ys)
        reporter.report({'loss': loss.data}, self)

        # To predict labels, call argmax method.
        _, predict = self.crf.argmax(hs)
        correct = 0
        total = 0
        for y, p in six.moves.zip(ys, predict):
            correct += self.xp.sum(y.data == p)
            total += len(y.data)
        reporter.report({'correct': correct}, self)
        reporter.report({'total': total}, self)

        return loss

    def argmax(self, xs):
        hs = [self.feature(x) for x in xs]
        return self.crf.argmax(hs)


class BiLSTMTagger(chainer.Chain):

    def __init__(self, n_dim, n_vocab, n_pos, n_hidden):
        super(BiLSTMTagger, self).__init__()
        with self.init_scope():
            self.embedid = L.EmbedID(in_size=n_vocab, out_size=n_dim)
            self.nstep_bilstm = L.NStepBiLSTM(n_layers=1, in_size=n_dim,
                                              out_size=n_hidden, dropout=0.0)
            self.linear = L.Linear(in_size=2*n_hidden, out_size=n_pos)

    def __call__(self, xs, ys):
        pred_ys = self.traverse(xs)
        loss = .0

        for pred_y, y in zip(pred_ys, ys):
            loss_ = F.softmax_cross_entropy(pred_y, y)
            loss += loss_

        reporter.report({'loss': loss.data}, self)
        return loss

    def traverse(self, xs):
        fs = []
        for x in xs:
            f = self.embedid(x)
            fs.append(f)

        hx, cx = None, None
        hx, cx, ys = self.nstep_bilstm(xs=fs, hx=hx, cx=cx)
        return [self.linear(y) for y in ys]

    def predict(self, xs):
        pred_ys = self.traverse(xs)
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=1) for pred_y in pred_ys]

        return pred_ys


class BiLSTMCRFTagger(chainer.Chain):

    def __init__(self, n_dim, n_vocab, n_pos, n_hidden):
        super(BiLSTMCRFTagger, self).__init__()
        with self.init_scope():
            self.embedid = L.EmbedID(in_size=n_vocab, out_size=n_dim)
            self.nstep_bilstm = L.NStepBiLSTM(n_layers=1, in_size=n_dim,
                                              out_size=n_hidden, dropout=0.0)
            self.linear = L.Linear(in_size=2*n_hidden, out_size=n_pos)
            self.crf = L.CRF1d(n_pos)

    def __call__(self, xs, ys):
        inds = numpy.argsort([-len(y) for y in ys]).astype('i')
        xs = [xs[i] for i in inds]
        ys = [ys[i] for i in inds]  # ys[inds]

        hs = self.traverse(xs)

        hs = F.transpose_sequence(hs)
        ys = F.transpose_sequence(ys)

        loss = self.crf(hs, ys)
        _, predict = self.crf.argmax(hs)

        reporter.report({'loss': loss.data}, self)
        return loss

    def traverse(self, xs):
        fs = []
        for x in xs:
            f = self.embedid(x)
            fs.append(f)

        hx, cx = None, None
        hx, cx, ys = self.nstep_bilstm(xs=fs, hx=hx, cx=cx)
        return [self.linear(y) for y in ys]

    def predict(self, xs):
        pred_ys = self.traverse(xs)
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=1) for pred_y in pred_ys]

        return pred_ys


# NOTE: This is under constructed
class BiLSTM_CNN_CRFTagger(chainer.Chain):

    def __init__(self, n_dim, n_vocab, n_pos, n_hidden):
        super(BiLSTMCNNCRFTagger, self).__init__()
        with self.init_scope():
            self.embedid = L.EmbedID(in_size=n_vocab, out_size=n_dim)
            self.nstep_bilstm = L.NStepBiLSTM(n_layers=1, in_size=n_dim,
                                              out_size=n_hidden, dropout=0.0)
            self.linear = L.Linear(in_size=2*n_hidden, out_size=n_pos)
            self.crf = L.CRF1d(n_pos)

    def __call__(self, xs, ys):
        inds = numpy.argsort([-len(y) for y in ys]).astype('i')
        xs = [xs[i] for i in inds]
        ys = [ys[i] for i in inds]  # ys[inds]

        hs = self.traverse(xs)

        hs = F.transpose_sequence(hs)
        ys = F.transpose_sequence(ys)

        loss = self.crf(hs, ys)
        _, predict = self.crf.argmax(hs)

        reporter.report({'loss': loss.data}, self)
        return loss

    def traverse(self, xs):
        fs = []
        for x in xs:
            f = self.embedid(x)
            fs.append(f)

        hx, cx = None, None
        hx, cx, ys = self.nstep_bilstm(xs=fs, hx=hx, cx=cx)
        return [self.linear(y) for y in ys]

    def predict(self, xs):
        pred_ys = self.traverse(xs)
        pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
        pred_ys = [pred_y.data.argmax(axis=1) for pred_y in pred_ys]

        return pred_ys
