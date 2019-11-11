<div align="center"><img src="./static/image/pyner.png" width="600"/></div>


# PyNER: Toolkit for sequence labeling in Chainer

[![CircleCI](https://circleci.com/gh/himkt/pyner.svg?style=svg)](https://circleci.com/gh/himkt/pyner)
[![GitHub stars](https://img.shields.io/github/stars/himkt/pyner.svg?maxAge=2592000&colorB=blue)](https://github.com/himkt/pyner/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/himkt/pyner.svg?colorB=red)](https://github.com/himkt/pyner/issues)
[![GitHub release](https://img.shields.io/github/release/himkt/pyner.svg?maxAge=2592000&colorB=yellow)](https://github.com/himkt/pyner)
[![MIT License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](LICENSE)

PyNER is a sequence labeling toolkit that allows researcher and developer to
train/evaluate neural sequence labeling methods.


# QuickStart

You can try `pyner` on a local machine or a docker container.

## 1. Local Machine

- setup (If you do not install [pipenv](https://github.com/pypa/pipenv), please install)

```
pipenv install
```

- train

```
# If a GPU is not available, specify `--gpu -1`
pipenv run python pyner/named_entity/train.py config/training/conll2003.lample.yaml --device 0
```

## 2. Docker container

- build container

```
make build
make build SUDO=  # if you are not administrator
```

- launch container

```
make start
make start SUDO=  # if you are not administrator
```

- train

You have to execute this command in Docker container.

```
# If a GPU is not available, specify `--gpu -1`
python3 train.py config/training/conll2003.lample.yaml --device 0
```

This experiment uses CoNLL 2003 dataset.
Please read the following "Prepare dataset" section.


# Prepare dataset

We use a data format same as [deep-crf](https://github.com/aonotas/deep-crf).

```
$ head -n 15 data/processed/CoNLL2003_BIOES/train.txt
EU      S-ORG
rejects O
German  S-MISC
call    O
to      O
boycott O
British S-MISC
lamb    O
.       O

Peter   B-PER
Blackburn       E-PER

BRUSSELS        S-LOC
1996-08-22      O
```

For reproducing results in [Lample's paper](https://aclweb.org/anthology/N16-1030),
you have to do some step to prepare datasets.

## 1. Prepare CoNLL 2003 Dataset

We can't include CoNLL 2003 dataset in this repository due to legal limitation.
Instead, PyNER offers the way to create dataset from CoNLL 2003 dataset

If you could prepare CoNLL 2003 dataset, you would have three files like below.

- eng.iob.testa
- eng.iob.testb
- eng.iob.train

Please put them to on same directoy (e.g. `data/external/conll2003`).

```
$ tree data/external/conll2003
data/external/conll2003
├── eng.iob.testa
├── eng.iob.testb
└── eng.iob.train
```

Then, you can create the dataset for pyner by following command.
After running the command, `./data/processed/CoNLL2003_BIOES` will be generated for you.

```
$ python bin/parse_CoNLL2003.py \
  --data-dir     data/external/conll2003 \
  --output-dir   data/processed/CoNLL2003_BIOES \
  --convert-rule iob2bioes
2019-09-24 23:43:39,299 INFO root :create dataset for CoNLL2003
2019-09-24 23:43:39,299 INFO root :create corpus parser
2019-09-24 23:43:39,300 INFO root :parsing corpus for training
2019-09-24 23:44:02,240 INFO root :parsing corpus for validating
2019-09-24 23:44:04,397 INFO root :parsing corpus for testing
2019-09-24 23:44:06,507 INFO root :Create train dataset
2019-09-24 23:44:06,705 INFO root :Create valid dataset
2019-09-24 23:44:06,755 INFO root :Create test dataset
2019-09-24 23:44:06,800 INFO root :Create vocabulary
$
$ tree data/processed/CoNLL2003_BIOES
data/processed/CoNLL2003_BIOES
├── test.txt
├── train.txt
├── valid.txt
├── vocab.chars.txt
├── vocab.tags.txt
└── vocab.words.txt
```


## 2. Prepare pre-trained Word Embeddings used in Lample's paper

Using pre-trained word embeddings significantly improve the performance of NER.
Lample et al. also use pre-trained word embeddings.
They use Skip-N-Gram embeddings, which can be downloaded from [Official repo's issue].
To use this, please run `make get-lample` before running `make build`.
(If you want to use GloVe embeddings, please run `make get-glove`.)

```
$ make get-lample
rm -rf data/external/GloveEmbeddings
mkdir -p data/external/LampleEmbeddings
mkdir -p data/processed/LampleEmbeddings
python bin/fetch_lample_embedding.py
python bin/prepare_embeddings.py \
                data/external/LampleEmbeddings/skipngram_100d.txt \
                data/processed/LampleEmbeddings/skipngram_100d \
                --format word2vec
saved model
$
$ ls -1 data/processed/LampleEmbeddings
skipngram_100d
skipngram_100d.vectors.npy
```

Congratulations! All preparation steps have done.
Now you can train the Lample's LSTM-CRF.
Please run the command:
- Local machine: `python3 pyner/named_entity/train.py config/training/conll2003.lample.yaml --device 0`
- Docker container: `python3 train.py config/training/conll2003.lample.yaml --device 0`


# Inference and Evaluate

You can test your model using `pyner/named_entity/inference.py`.
Only thing you have to pass to `inference.py` is path to model dir.
Model dir is defined in config file (**output**).

```
$ cat config/training/conll2003.lample.yaml
iteration: "./config/iteration/long.yaml"
external: "./config/external/conll2003.yaml"
model: "./config/model/lample.yaml"
optimizer: "./config/optimizer/sgd_with_clipping.yaml"
preprocessing: "./config/preprocessing/znorm.yaml"
output: "./model/conll2003.lample"  # model dir is here!!
```

If you successfully train the model, some files are generated on `model/conll2003.lample.skipngram.YYYY-MM-DDTxx:xx:xx.xxxxxx`.

```
$ ls -1 model/conll2003.lample.skipngram.2019-09-24T07:02:33.536822
args               
log                
snapshot_epoch_0001
snapshot_epoch_0002
snapshot_epoch_0003
snapshot_epoch_0004
...
snapshot_epoch_0148
snapshot_epoch_0149
snapshot_epoch_0150
validation.main.fscore.epoch_031.pred  # here!!
```

Running `python3 pyner/named_entity/inference.py` will generate prediction results on `model/conll2003.lample.skipngram.YYYY-MM-DDTxx:xx:xx.xxxxxx`
A file name would be `{metrics}.epoch_{xxx}.pred`.
`inference.py` check a log and select a model which achieve most high f1 score on development set.
You can use other selection criteria such as watching loss value and specifying an epoch.

- Dev loss: `python3 pyner/named_entity/inference.py --metrics validation/main/loss model/conll2003.lample.skipngram.2019-09-24T07:02:33.536822`)
- Specific epoch: `python3 pyner/named_entity/inference.py --epoch 1 model/conll2003.lample.skipngram.2019-09-24T07:02:33.536822`

If you could generate a prediction file, it's time to evaluate a model performance.
[conlleval](https://www.clips.uantwerpen.be/conll2000/chunking/output.html) is the standard script to evaluate CoNLL Chunking/NER tasks.
First of all, we have to download `conlleval`.
Running the command `make get-conlleval` would download `conlleval` on current directory.
Then, evaluate!!!

```
$ ./conlleval < model/conll2003.lample.skipngram.2019-09-24T07:02:33.536822/validation.main.fscore.epoch_139.pred
processed 46435 tokens with 5628 phrases; found: 5651 phrases; correct: 5134.
accuracy:  97.82%; precision:  90.85%; recall:  91.22%; FB1:  91.04
              LOC: precision:  93.41%; recall:  92.18%; FB1:  92.79  1640
             MISC: precision:  80.66%; recall:  80.66%; FB1:  80.66  693
              ORG: precision:  88.72%; recall:  89.79%; FB1:  89.26  1676
              PER: precision:  94.76%; recall:  96.23%; FB1:  95.49  1642
```

F1 score on test set is 91.04, which is approximately the same as the result in Lample's paper! (90.94)


### Reference
- [Neural Architectures for Named Entity Recognition]
  - NAACL2016, Lample et al.


[Neural Architectures for Named Entity Recognition]: https://aclweb.org/anthology/N16-1030
[Official repo's issue]: https://github.com/glample/tagger/issues/44
[CoNLL 2003]: https://www.clips.uantwerpen.be/conll2003/ner/
