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
		--build-arg UID=$(USERID) \
		--file=docker/Dockerfile

start:
	$(SUDO) $(DOCKER) run \
		--volume $(PWD):/docker \
		-it $(TAG)

test:
	python -m unittest discover

lint:
	flake8 pyner

conlleval:
	curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
	chmod 777 conlleval

tmux:
	tmux -f .dotfiles/.tmux.conf

get-glove:
	cd data && wget http://nlp.stanford.edu/data/glove.6B.zip
	cd data && unzip glove.6B.zip
	cd data && rm glove.6B.zip
