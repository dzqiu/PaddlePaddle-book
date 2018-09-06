import paddle.fluid as fluid
import numpy
import sys
from functools import partial
import collections
import tarfile

def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)
    for l in f:
        for w in l.strip().split(): #删除前后端空格，并且切分单词，每个单词计数
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1
    return word_freq

def build_dict(data_path,min_word_freq=50):
    """
    构建字典
    """
    train_filename = './simple-examples/data/ptb.train.txt'
    test_filename  = './simple-examples/data/ptb.valid.txt'
    with tarfile.open(data_path) as tf:
        trainf = tf.extractfile(train_filename)
        testf = tf.extractfile(test_filename)
        word_freq = word_count(testf, word_count(trainf))
        if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
            del word_freq['<unk>']
        word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items()) #滤除掉小于min_word的单词
        word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0])) #排序，次数多优先，首字母次之
        words, _ = list(zip(*word_freq_sorted))
        word_idx = dict(zip(words, xrange(len(words))))      #构建字典，字典顺序与次序无关
        word_idx['<unk>'] = len(words)
    return word_idx
