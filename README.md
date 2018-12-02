<div align="center"><img src="./static/image/pyner.png" width="800"/></div>

# PyNER: Extensible Implementation for Neural Named Entity Recognizer
PyNER is a chainer implementation for neural named entity recognizer.
It is designed so that users can define arbitrary module (e.g. LSTM, CNN, ...).
PyNER also provides Dataset class and Evaluator class suited to sequence labeling tasks.

### Prepare dataset

Please see [README for corpus parser](tool/corpus/README.md) and [README for word vector](tool/vector/README) 
to prepare dataset.

Dataset format is the same as https://github.com/guillaumegenthial/tf_ner/tree/master/data/example

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
make install  # you have to run this command on docker container
python3 pyner/named_entity/train.py config/001.config --gpu 0
```

## Reference
- [Neural Architectures for Named Entity Recognition]
  - NAACL2016
  - Lample et al.


[Neural Architectures for Named Entity Recognition]: https://arxiv.org/abs/1603.01360
