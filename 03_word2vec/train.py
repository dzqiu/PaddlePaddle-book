import paddle.v2 as paddle
import paddle.fluid as fluid
import numpy
import sys
from functools import partial
import collections
import tarfile
import math
import os

EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 100


use_cuda = True # set to True if training with GPU
data_path = '/home/dzqiu/DataSet/simple-examples.tgz'

word_dict = build_dict(data_path)
dict_size = len(word_dict)

def reader_creator(data_path,filename,word_idx,n):
    def reader():
        with tarfile.open(data_path) as tf:
            f = tf.extractfile(filename)
            UNK = word_idx['<e>']
            for l in f:#按照每行读取
                assert n > -1, 'Invalid gram length'
                l = ['<s>'] + l.strip().split() + ['<e>']
                if len(l) >= n:
                    l = [word_idx.get(w, UNK) for w in l]
                    for i in range(n, len(l) + 1):
                        yield tuple(l[i - n:i])

    return reader

def inference_program(is_sparse):
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.layers.embedding(
        input=fourth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(
        input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word
def inference_embedding(is_sparse=True,layer=1):
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.layers.embedding(
        input=fourth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    if layer==1:
        return embed_first
    elif layer==2:
        return embed_second
    elif layer==3:
        return embed_third
    else: return embed_fourth

def train_program(is_sparse):

    predict_word = inference_program(is_sparse)
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.AdagradOptimizer(
        learning_rate=3e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(8e-4))

def train(use_cuda, train_program, params_dirname):
    

    train_filename = './simple-examples/data/ptb.train.txt'
    test_filename = './simple-examples/data/ptb.valid.txt'
    train_reader = paddle.batch(
        reader_creator(data_path,train_filename,word_dict,N), BATCH_SIZE)
    test_reader = paddle.batch(
        reader_creator(data_path,test_filename,word_dict,N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            outs = trainer.test(
                reader=test_reader,
                feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
            avg_cost = outs[0]
            if event.step % 10 == 0:
                print "Step %d: Average Cost %f" % (event.step, avg_cost)


            if avg_cost < 5.8:
                trainer.save_params(params_dirname)
                trainer.stop()

            if math.isnan(avg_cost):
                sys.exit("got NaN loss, training failed.")

    trainer = fluid.Trainer(
        train_func=train_program,
        # optimizer=fluid.optimizer.SGD(learning_rate=0.001),
        optimizer_func=optimizer_func,
        place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=1,
        event_handler=event_handler,
        feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])

def embedding_infer(use_cuda, inference_program, params_dirname=None) :
    
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() 
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)
   

    data1 = [[20]] 
    data2 = [[20]]  
    data3 = [[20]] 
    data4 = [[20]]  
    lod = [[1]]
    first_word = fluid.create_lod_tensor(data1, lod, place)
    second_word = fluid.create_lod_tensor(data2, lod, place)
    third_word = fluid.create_lod_tensor(data3, lod, place)
    fourth_word = fluid.create_lod_tensor(data4, lod, place)

    embeding_layer = inferencer.infer(
        {
            'firstw': first_word,
            'secondw': second_word,
            'thirdw': third_word,
            'fourthw': fourth_word
        },
        return_numpy=False)

    print numpy.array(embeding_layer[0])
    

def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)


    

    data1 = [[word_dict['among']]] # 'among'
    data2 = [[word_dict['a']]]  # 'a'
    data3 = [[word_dict['group']]]  # 'group'
    data4 = [[word_dict['of']]]  # 'of'
    lod = [[1]]

    first_word = fluid.create_lod_tensor(data1, lod, place)
    second_word = fluid.create_lod_tensor(data2, lod, place)
    third_word = fluid.create_lod_tensor(data3, lod, place)
    fourth_word = fluid.create_lod_tensor(data4, lod, place)

    result = inferencer.infer(
        {
            'firstw': first_word,
            'secondw': second_word,
            'thirdw': third_word,
            'fourthw': fourth_word
        },
        return_numpy=False)
    print('softmax result=')
    print(numpy.array(result[0]))
#     print(numpy.array(embedding_second))
    most_possible_word_index = numpy.argmax(result[0])
#     print(most_possible_word_index)
    print('amog a group of :')
    print([
        key for key, value in word_dict.iteritems()
        if value == most_possible_word_index
    ][0])


def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    params_dirname = "word2vec.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=partial(train_program, is_sparse),
        params_dirname=params_dirname)

    infer(
        use_cuda=use_cuda,
        inference_program=partial(inference_program, is_sparse),
        params_dirname=params_dirname)
    print('first word embeding result:')
    embedding_infer(
        use_cuda=use_cuda,
        inference_program=partial(inference_embedding,is_sparse,1),
        params_dirname=params_dirname)
    print('second word embeding result:')
    embedding_infer(
        use_cuda=use_cuda,
        inference_program=partial(inference_embedding,is_sparse,2),
        params_dirname=params_dirname)


if __name__ == '__main__':
    main(use_cuda=use_cuda, is_sparse=True)
