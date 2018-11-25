FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
ENV PYTHONIOENCODING "utf-8"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        language-pack-ja \
        libssl-dev \
        zlib1g-dev \
        python3 \
        python3-pip \
        python3-setuptools \
        openssl \
        neovim \
        unzip \
        curl \
        wget \
        tmux \
        sudo \
        git \
        gcc \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:greymd/tmux-xpanes && \
    apt-get update && \
    apt-get install -y tmux-xpanes

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY requirements_gpu.txt /tmp/requirements_gpu.txt
RUN pip3 install -r /tmp/requirements_gpu.txt

RUN pip3 install -U git+https://github.com/himkt/chainer.git@add-permutate-in-crf1d

RUN curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > /tmp/conlleval
RUN chmod 777 /tmp/conlleval

WORKDIR /docker
