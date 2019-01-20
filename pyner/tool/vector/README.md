# Skip-gram: word embeddings for PyNER

### Get unlabeled large corpora

```
make get-jawiki
make get-enwiki
```

### Word Segmentation

For Japanese Corpus, it is needed to be segmented sentences into words.
You need to install `tiny_tokenizer`, which is installed by `pip install tiny_tokenizer`
TinyTokenizer uses MeCab or KyTea as a segmentor, so you have to install respectively.

```
make process-jawiki
make process-enwiki
```

### Training


### Use Lample's Embeddings

[Lample's SkipNGram] is used in original paper.
Download embedding from google-drive indicated in [Official repo's issue].
Download and rename to `lample_embedding.txt`
Then please run a following command.

```
python vector/word2vec2gensim.py ./lample_embedding.txt data/skip_ngram_lample+dimension_100+window_8
```

[Official repo's issue]: https://github.com/glample/tagger/issues/44
