export DOCKER=nvidia-docker
export TAG='himkt/pyner:latest'
export PWD=`pwd`
export USERID=`id -u`
export USERGROUPID=`id -g`
export SUDO='sudo'

.PHONY: build start test lint

build:
	$(SUDO) $(DOCKER) build -t $(TAG) . \
		--build-arg UID=$(USERID) \
		--file=docker/Dockerfile.rootless

start:
	$(SUDO) $(DOCKER) run -it \
		--volume $(PWD):/docker \
		--user $(USERID):$(USERGROUPID) $(TAG)

build_rootless:
	$(DOCKER) build -t $(TAG) . \
		--file=docker/Dockerfile

start_rootless:
	$(DOCKER) run -it \
		--volume $(PWD):/docker $(TAG)

test:
	python -m unittest discover

lint:
	flake8 pyner

install:
	python3.6 setup.py install --user

conll:
	cd ./tool && curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
	cd ./tool && chmod 777 conlleval

tmux:
	tmux -f .dotfiles/.tmux.conf
