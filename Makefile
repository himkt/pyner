export DOCKER=nvidia-docker
export TAG='himkt/pyner:latest'
export PWD=`pwd`
export USERID=`id -u`
export USERGROUPID=`id -g`
export SUDO='sudo'


.PHONY: build start test lint

build:
	$(SUDO) $(DOCKER) build \
		-t $(TAG) . \
		--build-arg GID=$(USERGROUPID) \
		--build-arg UID=$(USERID)

start:
	$(SUDO) $(DOCKER) run \
		--user $(USERID):$(USERID) \
		--volume $(PWD)/data:/home/docker/data \
		--volume $(PWD)/model:/home/docker/model \
		-it $(TAG)

test:
	pipenv run python -m unittest discover

lint:
	flake8 pyner

get-glove:
	mkdir -p data/external/GloveEmbeddings
	mkdir -p data/processed/GloveEmbeddings
	cd data/external/GloveEmbeddings && wget http://nlp.stanford.edu/data/glove.6B.zip
	cd data/external/GloveEmbeddings && unzip glove.6B.zip
	cd data/external/GloveEmbeddings && rm glove.6B.zip
	pipenv run python bin/prepare_embeddings.py \
		data/external/GloveEmbeddings/glove.6B.100d.txt \
		data/processed/GloveEmbeddings/glove.6B.100d \
		--format glove

get-lample:
	rm -rf data/external/GloveEmbeddings
	mkdir -p data/external/LampleEmbeddings
	mkdir -p data/processed/LampleEmbeddings
	pipenv run python bin/fetch_lample_embedding.py
	pipenv run python bin/prepare_embeddings.py \
			data/external/LampleEmbeddings/skipngram_100d.txt \
			data/processed/LampleEmbeddings/skipngram_100d \
			--format word2vec

get-conlleval:
	curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
	chmod 777 conlleval
