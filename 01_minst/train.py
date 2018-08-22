#coding:utf-8
import os
import paddle
from PIL import Image
import paddle.fluid as fluid
import numpy
import platform
import subprocess

def reader_creator(image_filename,label_filename,buffer_size):
    def reader():
	#调用命令读取文件，Linux下使用zcat
        if platform.system()=='Linux':
            zcat_cmd = 'zcat'
        elif paltform.system()=='Windows':
            zcat_cmd = 'gzcat'
        else:
            raise NotImplementedError("This program is suported on Windows or Linux,\
                                      but your platform is" + platform.system())
        
        #create a subprocess to read the images
        sub_img = subprocess.Popen([zcat_cmd, image_filename], stdout = subprocess.PIPE)
        sub_img.stdout.read(16) #skip some magic bytes 这里我们已经知道，所以我们不在需要前16字节
        #create a subprocess to read the labels
        sub_lab = subprocess.Popen([zcat_cmd, label_filename], stdout = subprocess.PIPE)
        sub_lab.stdout.read(8)  #skip some magic bytes 同理
        
	try:
            while True:         #前面使用try,故若再读取过程中遇到结束则会退出
		#label is a pixel repersented by a unsigned byte,so just read a byte
                labels = numpy.fromfile(
                            sub_lab.stdout,'ubyte',count=buffer_size).astype("int")

                if labels.size != buffer_size:
                    break
		#read 28*28 byte as array,and then resize it
                images = numpy.fromfile(
                            sub_img.stdout,'ubyte',count=buffer_size * 28 * 28).reshape(
                                buffer_size, 28, 28).astype("float32")
		#mapping each pixel into (-1,1)
                images = images / 255.0 * 2.0 - 1.0;
                for i in xrange(buffer_size):
                    yield images[i,:],int(labels[i]) #将图像与标签抛出，循序与feed_order对应！
        finally:
            try:
		#terminate the reader subprocess
                sub_img.terminate()
            except:
                pass
            try:
		#terminate the reader subprocess
                sub_lable.terminate()
            except:
                pass
    return reader

#a full-connect-layer network using softmax as activation function
def softmax_regression():
    img = fluid.layers.data(name='img',shape=[1,28,28],dtype='float32')
    predict = fluid.layers.fc(input=img,size=10,act='softmax')
    return predict
#3 full-connect-layers network using softmax as activation function
def multilayer_perceptron():
    img = fluid.layers.data(name='img',shape=[1,28,28],dtype='float32')
    hidden = fluid.layers.fc(input = img,size=128,act='softmax')
    hidden = fluid.layers.fc(input = hidden,size=64,act='softmax')
    prediction = fluid.layers.fc(input = hidden,size=10,act='softmax')
    return prediction
#traditional converlutional neural network
def cnn():
    img = fluid.layers.data(name='img',shape=[1, 28, 28], dtype ='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input = img,
        filter_size = 5,
        num_filters = 20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # second conv pool
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # output layer with softmax activation function. size = 10 since there are only 10 possible digits.
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction

def train_program():
    #if using dtype='int64', it reports errors!
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Here we can build the prediction network in different ways. Please
    predict = cnn()
    #predict = softmax_regression()
    #predict = multilayer_perssion()

    # Calculate the cost from the prediction and label.
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

if __name__ == "__main__":
    print("run minst train\n")
    minst_prefix = '/home/dzqiu/DataSet/minst/'
    train_image_path   = minst_prefix + 'train-images-idx3-ubyte.gz'
    train_label_path   = minst_prefix + 'train-labels-idx1-ubyte.gz'
    test_image_path    = minst_prefix + 't10k-images-idx3-ubyte.gz'
    test_label_path    = minst_prefix + 't10k-labels-idx1-ubyte.gz'
    #reader_creator在将在下面讲述
    train_reader = paddle.batch(paddle.reader.shuffle(#shuffle用于打乱buffer的循序
					reader_creator(train_image_path,train_label_path,buffer_size=100),
                                        buf_size=500),
                                        batch_size=64)
    test_reader  = paddle.batch(reader_creator(test_image_path,test_label_path,buffer_size=100),
			batch_size=64)		      #测试集就不用打乱了
    
    #if use GPU, use 'export FLAGS_fraction_of_gpu_memory_to_use=0' at first
    use_cuda = True
    place    = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    
    trainer  = fluid.Trainer(train_func=train_program,place=place,optimizer_func=optimizer_program)
    
    params_dirname = "recognize_digits_network.inference.model"
    lists = []
    #
    def event_handler(event):
        if isinstance(event,fluid.EndStepEvent):#每步触发事件
            if event.step % 100 == 0:
                print("Pass %d, Epoch %d, Cost %f, Acc %f"\
                       %(event.step, event.epoch,
                       event.metrics[0],#这里就是train_program返回的第一个参数arg_cost（当然也就是第零个）
                       event.metrics[1]))#这里是train_program返回的第二个参数acc
 
        if isinstance(event,fluid.EndEpochEvent):#每次迭代触发事件
	    trainer.save_params(params_dirname)
            #使用test的时候，返回值就是train_program的返回，所以赋值需要对应
            avg_cost, acc = trainer.test(reader=test_reader,feed_order=['img','label']) 
            print("Test with Epoch %d, avg_cost: %s, acc: %s"
                  %(event.epoch, avg_cost, acc))
            lists.append((event.epoch, avg_cost, acc))

    # Train the model now
    trainer.train(num_epochs=5,event_handler=event_handler,reader=train_reader,feed_order=['img', 'label'])
    
    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (float(best[2]) * 100)

    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32) #[N C H W] 这里多了一个N
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/infer_3.png')
    inferencer = fluid.Inferencer(
        # infer_func=softmax_regression, # uncomment for softmax regression
        # infer_func=multilayer_perceptron, # uncomment for MLP
        infer_func=cnn,  # uncomment for LeNet5
        param_path=params_dirname,
        place=place)

    results = inferencer.infer({'img': img})
    lab = numpy.argsort(results)  # probs and lab are the results of one batch data
    print "Label of infer_3.png is: %d" % lab[0][0][-1]
