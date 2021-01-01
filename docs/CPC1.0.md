# Cookpad Parsed Corpus v1.0

You can train a BiLSTM-CRF model used in the paper of Cookpad Parsed Corpus (CPC1.0).
Note that you have to run `make conll2003` on the [CPC1.0 repository](https://github.com/cookpad/cpc1.0) in advance.


## Put CPC1.0

```sh
mv `path_to_CPC_repository`/outputs/conll2003 data/external/CPC1.0
```


## Create dataset

```sh
poetry run python bin/parse_CoNLL2003.py \
    --delimiter "\t" \
    --data-dir data/external/CPC1.0 \
    --output-dir data/processed/CPC1.0
```


## Build an image

```sh
docker build -t himkt/pyner .
```


## Train a model

```sh
docker run \
    -v `pwd`/model:/work/model \
    --gpus all \
    -it \
    --rm himkt/pyner \
    poetry run python pyner/named_entity/train.py \
        config/training/cpc1.0.lample.yaml \
        --device 0 \
        --seed 24
```


## Inference

```sh
docker run \
    -v `pwd`/model:/work/model \
    --gpus all \
    -it \
    --rm himkt/pyner \
    poetry run python pyner/named_entity/inference.py \
        --metric validation/main/fscore \
        model/cpc1.0.lample.`timestamp_of_your_model`
```


# Reference

- Harashima and Hiramatsu, [Cookpad Parsed Corpus: Linguistic Annotations of Japanese Recipes](https://www.aclweb.org/anthology/2020.law-1.8/), LAW, 2020.
