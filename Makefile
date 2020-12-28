export TAG='himkt/pyner:latest'


.PHONY: build start test lint

build:
	docker build -t $(TAG) .

start:
	docker run --gpus all --rm -it $(TAG)

test:
	poetry run python -m pytest tests

lint:
	poetry run flake8 pyner

get-glove:
	mkdir -p data/external/GloveEmbeddings
	mkdir -p data/processed/GloveEmbeddings
	cd data/external/GloveEmbeddings && wget http://nlp.stanford.edu/data/glove.6B.zip
	cd data/external/GloveEmbeddings && unzip glove.6B.zip
	cd data/external/GloveEmbeddings && rm glove.6B.zip
	poetry run python bin/prepare_embeddings.py \
		data/external/GloveEmbeddings/glove.6B.100d.txt \
		data/processed/GloveEmbeddings/glove.6B.100d \
		--format glove

get-lample:
	rm -rf data/external/GloveEmbeddings
	mkdir -p data/external/LampleEmbeddings
	mkdir -p data/processed/LampleEmbeddings
	poetry run python bin/fetch_lample_embedding.py
	poetry run python bin/prepare_embeddings.py \
			data/external/LampleEmbeddings/skipngram_100d.txt \
			data/processed/LampleEmbeddings/skipngram_100d \
			--format word2vec

get-conlleval:
	curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
	chmod 777 conlleval
