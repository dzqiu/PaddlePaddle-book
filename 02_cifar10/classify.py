#coding:utf-8
from __future__ import print_function
import paddle.fluid as fluid
import paddle
import numpy as np 
import sys
import os
import cPickle as Pickle 

cifar_classes = ['air plane','automobile','bird','cat','deer','dog','frog','house','ship','truck']

def reader_creator(ROOT,istrain=True,cycle=False):
    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename,'rb') as f:
            datadict = Pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            """ (N C H W) transpose to (N H W C) """
            X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
            Y = np.array(Y)
            return X,Y
    def reader():
        while True:
            if istrain:
                for b in range(1,6):
                    f   = os.path.join(ROOT,'data_batch_%d'%(b))
                    X,Y = load_CIFAR_batch(f)
                    length = X.shape[0]
                    for i in range(length):
                        yield X[i],Y[i]
                if not cycle:
                    break
            else:
                f = os.path.join(ROOT,'test_batch')
                X,Y = load_CIFAR_batch(f)
                length = X.shape[0]
                for i in range(length):
                    yield X[i],Y[i]
                if not cycle:
                    break
    return reader

def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            #一组的卷积层的卷积核总数,组成list[num_filter num_filter ...]
            conv_num_filter=[num_filter] * groups, 
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            #每组卷积层各层的droput概率
            conv_batchnorm_drop_rate=dropouts, 
            pool_size=2,
            pool_stride=2,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0]) #[0.3 0]即为第一组两层的dorpout概率，下同
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)

    bn = fluid.layers.batch_norm(input=fc1, act='relu')

    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)

    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict




def train_network():
    predict = inference_network()
    label = fluid.layers.data(name='label',shape=[1],dtype='int64')
    cost  = fluid.layers.cross_entropy(input=predict,label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict,label=label)
    return [avg_cost,accuracy]

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

def train(data_path,save_path):
    BATCH_SIZE = 128
    EPOCH_NUM  = 50
    train_reader = paddle.batch(
        paddle.reader.shuffle(reader_creator(data_path),buf_size=50000),
        batch_size = BATCH_SIZE)
    test_reader  = paddle.batch(
        reader_creator(data_path,False),
        batch_size=BATCH_SIZE)
    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 100 == 0:
                print("\nPass %d, Epoch %d, Cost %f, Acc %f" %
                      (event.step, event.epoch, event.metrics[0],
                       event.metrics[1]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, fluid.EndEpochEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=['image', 'label'])
            print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                event.epoch, avg_cost, accuracy))
            if save_path is not None:
                trainer.save_params(save_path)
    place = fluid.CUDAPlace(0) 
    trainer = fluid.Trainer(
        train_func=train_network, optimizer_func=optimizer_program, place=place)
    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=['image', 'label'])

def inference_network():
    """" The image is 32*32 with RGB representation"""
    data_shape = [3,32,32]
    image = fluid.layers.data(name='image',shape=data_shape,dtype='float32')
    predict = vgg_bn_drop(image)
    return predict
def infer(params_dir):
    place = fluid.CUDAPlace(0)
    inferencer = fluid.Inferencer(
        infer_func=inference_network, param_path=params_dir, place=place)
     # Prepare testing data. 
    from PIL import Image
    import numpy as np
    import os

    def load_image(file):
        im = Image.open(file)
        im = im.resize((32, 32), Image.ANTIALIAS)
        im = np.array(im).astype(np.float32)
        """transpose [H W C] to [C H W]"""
        im = im.transpose((2, 0, 1)) 
        im = im / 255.0

        # Add one dimension, [N C H W] N=1
        im = np.expand_dims(im, axis=0)
        return im
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/dog.png')
    # inference
    results = inferencer.infer({'image': img})
    print(results)
    lab = np.argsort(results)  # probs and lab are the results of one batch data
    print("infer results: ", cifar_classes[lab[0][0][-1]])


if __name__ == '__main__':
    print("Classify the cifar10 images...")
    data_path = '/home/dzqiu/DataSet/cifar-10-batches-py/'
    save_path = 'image_classification_resnet.inference.model'
    train(data_path,save_path)
    infer(save_path)

    