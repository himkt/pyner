FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONIOENCODING UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential language-pack-ja \
        python3 python3-dev python3-pip python3-setuptools \
        curl tmux sudo git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pip --upgrade
RUN pip3 install setuptools --upgrade
RUN pip3 install poetry

WORKDIR /work
COPY pyproject.toml .
COPY poetry.lock .
COPY pyner ./pyner
RUN  poetry install

WORKDIR /tmp
RUN curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
RUN chmod 777 conlleval

WORKDIR /work
COPY data   ./data
COPY config ./config

COPY bin ./bin
