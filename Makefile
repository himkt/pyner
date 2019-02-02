export DOCKER=nvidia-docker
export TAG='himkt/pyner:latest'
export PWD=`pwd`
export USERID=`id -u`
export USERGROUPID=`id -g`


.PHONY: build start test lint

build:
	sudo build -t $(TAG) . \
		--build-arg UID=$(USERID) \
		--file=docker/Dockerfile

start:
	sudo $(DOCKER) run -it \
		--volume $(PWD):/docker

build_rootless:
	$(DOCKER) build -t $(TAG) . \
		--build-arg UID=$(USERID) \
		--file=docker/Dockerfile.rootless

start_rootless:
	$(DOCKER) run -it \
		--volume $(PWD):/docker \
		--user $(USERID):$(USERGROUPID) $(TAG)

test:
	python -m unittest discover

lint:
	flake8 pyner

conlleval:
	cd ./pyner/tool && curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
	cd ./pyner/tool && chmod 777 conlleval

tmux:
	tmux -f .dotfiles/.tmux.conf

get-glove:
	cd data && wget http://nlp.stanford.edu/data/glove.6B.zip
	cd data && unzip glove.6B.zip
	cd data && rm glove.6B.zip
