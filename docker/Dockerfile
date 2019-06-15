# Dockerfile for himkt/pyner


FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV PYTHONIOENCODING "utf-8"
ARG GID
ARG UID


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential language-pack-ja \
        python3 python3-pip python3-setuptools \
        curl tmux sudo git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:greymd/tmux-xpanes && \
    apt-get update && \
    apt-get install -y tmux-xpanes


RUN pip3 install pip --upgrade
RUN pip3 install setuptools --upgrade


WORKDIR /tmp/library

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt -U

COPY requirements_gpu.txt requirements_gpu.txt
RUN pip3 install -r requirements_gpu.txt -U

COPY setup.py setup.py
COPY pyner pyner
RUN pip3 install .


WORKDIR /tmp
RUN curl https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt > conlleval
RUN chmod 777 conlleval


RUN groupadd -g $GID dockerg && \
    useradd -u $UID -g dockerg -G sudo -m -s /bin/bash docker && \
    echo 'docker:root' | chpasswd

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


WORKDIR /home/docker

COPY data data
COPY config config
COPY pyner/named_entity/train.py train.py
COPY pyner/tool tool


USER docker
