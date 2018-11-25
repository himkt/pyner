# Parser for CoNLL formatted corpus

You have to prepare [CoNLL2003] dataset because it is not allowed to distribute.

### Processing

It assume that you have a directory which contains CoNLL formatted dataset.
Dataset structure is like below.

```
/datasets
├── eng.iob.testa
├── eng.iob.testb
└── eng.iob.train
```

You can create dataset for pyner by following command.

```
python corpus/parse_CoNLL2003.py --data-dir ./dataset
```

After running the command, `./data/processed/CoNLL2003` is generated for you!

[CoNLL2003]: https://www.clips.uantwerpen.be/conll2003/ner/
