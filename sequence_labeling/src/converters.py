import chainer
import numpy


def converter(batch, device):
    '''
    converter for word only dataset
    '''
    wordids, tagids, labelids = [], [], []
    for wordids_, _, tagids_, labelids_ in batch:
        if device > -1:
            wordids_ = chainer.cuda.to_gpu(wordids_)
            tagids_ = chainer.cuda.to_gpu(tagids_)
            labelids_ = chainer.cuda.to_gpu(labelids_)

        wordids.append(wordids_)
        tagids.append(tagids_)
        labelids.append(labelids_)

    wordids = numpy.asarray(wordids)
    tagids = numpy.asarray(tagids)
    labelids = numpy.asarray(labelids)
    lengthes = [b.size for b in wordids]
    index = numpy.argsort(lengthes)[::-1]

    wordids = wordids[index]
    tagids = tagids[index]
    labelids = labelids[index]

    return wordids, tagids, labelids


def converter2(batch, device):
    '''
    converter for word and char dataset
    '''
    wordids, charids_seq, tagids, labelids = [], [], [], []
    for wordids_, charids_seq_, tagids_, labelids_ in batch:
        if device > -1:
            wordids_ = chainer.cuda.to_gpu(wordids_)
            charids_seq_ = numpy.asarray([chainer.cuda.to_gpu(charids) for charids in charids_seq_])  # NOQA
            tagids_ = chainer.cuda.to_gpu(tagids_)
            labelids_ = chainer.cuda.to_gpu(labelids_)

        wordids.append(wordids_)
        charids_seq.append(charids_seq_)
        tagids.append(tagids_)
        labelids.append(labelids_)

    wordids = numpy.asarray(wordids)
    charids_seq = numpy.asarray(charids_seq)
    tagids = numpy.asarray(tagids)
    labelids = numpy.asarray(labelids)
    lengthes = [b.size for b in wordids]
    index = numpy.argsort(lengthes)[::-1]

    wordids = wordids[index]
    charids_seq = charids_seq[index]
    tagids = tagids[index]
    labelids = labelids[index]

    return wordids, charids_seq, tagids, labelids
