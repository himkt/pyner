#!/usr/bin/env python

from setuptools import find_packages, setup

try:
    import subprocess

    nvcc_version = subprocess.check_output("nvcc --version", shell=True)
    cuda_version = nvcc_version.decode("utf-8") \
        .split("\n")[-2] \
        .split(",")[-2]  \
        .split(" ")[-1]
    cuda_version = cuda_version.replace(".", "")  # 10.0 -> 100
    cupy_version = "cupy-cuda{}==7.0.0b2".format(cuda_version)
except Exception:
    cupy_version = None


install_requires = []

install_requires.append("gensim==3.7.3")
install_requires.append("numpy==1.17.0")
install_requires.append("scikit-learn==0.21.2")
install_requires.append("seqeval==0.0.5")
install_requires.append("chainer==7.0.0b2")
install_requires.append("chainerui==0.9.0")
install_requires.append("PyYAML==5.1.1")

extra_requires = []
extra_requires.append("tiny_tokenizer")

if cupy_version is not None:
    install_requires.append(cupy_version)

setup(
    name="pyner",
    version="1.0",
    description="Neural Named Entity Recognizer",
    author="himkt",
    author_email="himkt@klis.tsukuba.ac.jp",
    url="https://github.com/himkt/pyner",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "tagger": ["tiny_tokenizer"]
    },
)
