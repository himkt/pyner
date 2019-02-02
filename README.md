<div align="center"><img src="./static/image/pyner.png" width="800"/></div>


# PyNER: Extensible Implementation of Neural Named Entity Recognizer

PyNER is a chainer implementation for neural named entity recognizer.
It is designed so that users can define arbitrary module (e.g. LSTM, CNN, ...).
PyNER also provides Dataset class and Evaluator class suited to sequence labeling tasks.


### QuickStart

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

In Docker container

- run experiments

```
pip3 install . # you have to run this command on docker container
python3 pyner/named_entity/train.py config/001.config --gpu 0
```


### Prepare dataset

Dataset format is the same as https://github.com/guillaumegenthial/tf_ner/tree/master/data/example

#### Using CoNLL 2003 Dataset

PyNER offers the way to create dataset from CoNLL 2003 dataset
*You have to prepare [CoNLL2003] dataset because it is not allowed to distribute.*

If you could prepare CoNLL 2003 dataset, you would have three files like below.

─ eng.iob.testa
─ eng.iob.testb
─ eng.iob.train

Please put them to the same directoy (e.g. `datasets`).
Dataset structure is like below.

```
/datasets
├── eng.iob.testa
├── eng.iob.testb
└── eng.iob.train
```

Then, you can create the dataset for pyner by following command.

```
python pyner/tool/corpus/parse_CoNLL2003.py --data-dir datasets3 --output-dir data/processed/CoNLL2003_BIOES --format iob2bioes
```

After running the command, `./data/processed/CoNLL2003_BIOES` is generated for you!


#### Use Lample's Embeddings

Using pre-trained word embeddings significantly improve the performance of NER.
Lample et al. also use pre-trained word embeddings.
They use Skip-N-Gram embeddings, which can be downloaded from [Official repo's issue].
To use this, please download and rename to `lample_embedding.txt`
Then run a following command.

```
python pyner/tool/vector/word2vec2gensim.py ./lample_embedding.txt data/processed/lample_embeddings
```


### Reference
- [Neural Architectures for Named Entity Recognition]
  - NAACL2016
  - Lample et al.


[Neural Architectures for Named Entity Recognition]: https://arxiv.org/abs/1603.01360
[Official repo's issue]: https://github.com/glample/tagger/issues/44
[CoNLL2003]: https://www.clips.uantwerpen.be/conll2003/ner/
