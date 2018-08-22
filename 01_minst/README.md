## Prerequisites
This codebase was developed and tested with PaddlePaddle 0.14, CUDA 8.0 and Ubuntu 16.04.

## Running
You can install PaddlePaddle using:
```bash
pip install paddlepaddle
```
Download the minst dataset from [minst](http://yann.lecun.com/exdb/mnist/), totally 4 files:
```
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
```
Change the prefix in Line112, and choose a network in Line98~100, and run:
```bash
python train.py
```
