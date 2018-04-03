import numpy


class CoNLLParser:

    def __init__(self):
        self.char2index = None
        self.word2index = None
        self.postag2index = None
        self.chktag2index = None
        self.nertag2index = None
        self.result = {}
        self.max_charlen = 0

    def read_data(self, filename):
        charids_sequence = []
        wordids_sequence = []
        postagids_sequence = []
        chktagids_sequence = []
        nertagids_sequence = []

        if self.char2index is None:
            char2index = {'<PADDING>': 0, '<OOV>': 1}
        else:
            char2index = self.char2index

        if self.word2index is None:
            word2index = {'<BOS>': 0, '<EOS>': 1, '<OOV>': 2}
        else:
            word2index = self.word2index

        if self.postag2index is None:
            postag2index = {'<BOS>': 0, '<EOS>': 1}
        else:
            postag2index = self.postag2index

        if self.chktag2index is None:
            chktag2index = {'<BOS>': 0, '<EOS>': 1}
        else:
            chktag2index = self.chktag2index

        if self.nertag2index is None:
            nertag2index = {'<BOS>': 0, '<EOS>': 1}
        else:
            nertag2index = self.nertag2index

        charids, wordids, postagids, chktagids, nertagids = [], [], [], [], []
        for morph in open(filename):
            if morph.startswith('-DOCSTART-'):
                continue

            morph = morph.rstrip()
            elems = morph.split()

            if len(elems) == 0:
                if not wordids:
                    continue

                if len(charids) > 10:
                    charids_sequence.append(numpy.asarray(charids))
                    wordids_sequence.append(numpy.asarray(wordids, dtype=numpy.int32))  # NOQA
                    postagids_sequence.append(numpy.asarray(postagids, dtype=numpy.int32))  # NOQA
                    chktagids_sequence.append(numpy.asarray(chktagids, dtype=numpy.int32))  # NOQA
                    nertagids_sequence.append(numpy.asarray(nertagids, dtype=numpy.int32))  # NOQA
                charids, wordids, postagids, chktagids, nertagids = [], [], [], [], []  # NOQA

            if len(elems) == 4:
                word, postag, chktag, nertag = elems
                self.max_charlen = max(self.max_charlen, len(word))

                for char in list(word):
                    if char not in char2index:
                        char2index[char] = len(char2index)

                if word not in word2index:
                    word2index[word] = len(word2index)

                if postag not in postag2index:
                    postag2index[postag] = len(postag2index)

                if chktag not in chktag2index:
                    chktag2index[chktag] = len(chktag2index)

                if nertag not in nertag2index:
                    nertag2index[nertag] = len(nertag2index)

                charid_ = numpy.asarray([char2index[char] for char in list(word)], dtype=numpy.int32)  # NOQA
                wordid = word2index[word]
                postagid = postag2index[postag]
                chktagid = chktag2index[chktag]
                nertagid = nertag2index[nertag]

                charids.append(charid_)
                wordids.append(wordid)
                postagids.append(postagid)
                chktagids.append(chktagid)
                nertagids.append(nertagid)

        charids_sequence = numpy.asarray(charids_sequence)
        wordids_sequence = numpy.asarray(wordids_sequence)
        postagids_sequence = numpy.asarray(postagids_sequence)
        chktagids_sequence = numpy.asarray(chktagids_sequence)
        nertagids_sequence = numpy.asarray(nertagids_sequence)

        self.char2index = char2index
        self.word2index = word2index
        self.postag2index = postag2index
        self.chktag2index = chktag2index
        self.nertag2index = nertag2index
        self.result[filename] = {'char': charids_sequence,
                                 'word': wordids_sequence,
                                 'postag': postagids_sequence,
                                 'chktag': chktagids_sequence,
                                 'nertag': nertagids_sequence}

    def get_data(self, filename, attr):
        return self.result[filename][attr]

    def load_embedding(self, filename):
        read_fp = open(filename)
        header = read_fp.readline().rstrip()
        local_n_vocab, word_dim = map(int, header.split())

        local_syn0 = numpy.random.random((local_n_vocab, word_dim))
        local_word2index = {}

        local_indices = []
        global_indices = []

        for line in read_fp:
            line = line.rstrip()
            elems = line.split()

            word = ''.join(elems[:-word_dim])
            vec = list(map(float, elems[-word_dim:]))

            if word not in local_word2index:
                local_word2index[word] = len(local_word2index)

            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)

            local_wordid = local_word2index[word]
            local_syn0[local_wordid, :] = vec

            global_wordid = self.word2index[word]
            local_indices.append(local_wordid)
            global_indices.append(global_wordid)

        n_vocab = len(self.word2index)
        syn0 = numpy.random.random((n_vocab, word_dim))
        syn0[global_indices] = local_syn0[local_indices]
        return syn0


if __name__ == '__main__':
    parser = CoNLLParser()
    label_data = '../input/CoNLL-2003/eng.train'
    parser.read_data(label_data)
    print(parser.get_data(label_data, 'char'))
    print(parser.get_data(label_data, 'word'))

    syn0 = parser.load_embedding('../input/GloVe/glove.6B.50d.txt')
    print(list(parser.word2index.keys())[:100])
    print(parser.max_charlen)
