# Sequence Labeling Chainer

Implementations for sequence labeling model using chainer

# Available models

- CRF
- BiLSTM
- BiLSTM-CRF
- BiLSTM-CRF with character-based feature
- semi-supervised BiLSTM-CRF


# Evaluators

This repo provides a chainer evaluator for NER task

Please see src/evaluators.py in detail.


# Data Format

We use CoNLL2003 shared task dataset.
But training dataset can't be distributed, you have to create yourself.

Annotation data is free to download: https://www.clips.uantwerpen.be/conll2003/ner/.
Corpus data is also free but an application require: https://trec.nist.gov/data/reuters/reuters.html

In this implementations, algorithms use word, postag columns as features.

Please read the README in ner.tgz carefully.
If you succeeded to compile dataset, you would get dataset like below.

```
U.N.    NNP     I-NP    I-ORG
official        NN      I-NP    O
Ekeus   NNP     I-NP    I-PER
heads   VBZ     I-VP    O
for     IN      I-PP    O
Baghdad NNP     I-NP    I-LOC
```

# Usage

```
python src/train.py --result-dir ./20180312_conll_200d --word-vec input/GloVe/glove.6B.200d.txt --word-dim 200 --train input/CoNLL-2003/eng.train --test input/CoNLL-2003/eng.testa --gpu 1 --model bilstm_crf
```

- `result-dir` specify the directory of experimental results for [chainerUI](https://github.com/chainer/chainerui)
- `word-vec` specify the path to a pr-train word-vector (it requires the Word2Vec format: https://radimrehurek.com/gensim/scripts/glove2word2vec.html)
- `train` specify the path to a file for training
- `test` specify the path to a file for testing
- `model` specify the model you want to train
