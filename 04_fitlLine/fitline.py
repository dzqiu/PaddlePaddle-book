#coding:utf-8
import paddle.fluid as fluid
import paddle
import numpy as np
import  matplotlib.pyplot as plt
import time


train_x  = np.linspace(-1, 1, 128)       # shape (100, 1)
noise    = np.random.normal(0, 0.1, size=train_x.shape)
train_y  = np.power(train_x, 2) + noise   

def reader():
    def reader_creator():
        for i in range(128):   
            yield train_x[i],train_y[i]
    return reader_creator

train_reader = paddle.batch(reader(),batch_size=64)

input_layer   = fluid.layers.data(name='data',shape=[1],dtype='float32')
hid   = fluid.layers.fc(input=input_layer, size=10, act='relu')
output = fluid.layers.fc(input=hid, size=1, act=None)
label  = fluid.layers.data(name='label',shape=[1],dtype='float32')

#损失函数采用均方差
cost   = fluid.layers.square_error_cost(input=output,label=label)
avg_cost = fluid.layers.mean(cost)
#优化器选择
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01)
opts      = optimizer.minimize(avg_cost)


place = fluid.CPUPlace()
feeder = fluid.DataFeeder(place=place, feed_list=['data', 'label'])
test_program = fluid.default_main_program().clone(for_test=True)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

plt.figure(1)    
plt.ion()
for pass_id in range(1000):
    for batch_id,data in enumerate(train_reader()):
        loss = exe.run(fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
    x_ = []
    y_ = []
    l_= []
    for batch_id,data in enumerate(train_reader()):
        x,y,l=exe.run(program=test_program,
               feed=feeder.feed(data),
               fetch_list=[input_layer,output,label])

        x_ =np.hstack((x_,x.ravel()))
        y_ =np.hstack((y_,y.ravel()))
        l_ =np.hstack((l_,l.ravel()))

    plt.cla()
    plt.scatter(x_,l_)
    plt.plot(x_,y_,'r')
    plt.text(0.3, -0.38, 'epoch %d Loss=%.4f' % (pass_id,loss[0]), fontdict={'size': 10, 'color': 'red'})
    plt.pause(0.001)

    plt.savefig('plot%05d.png'%pass_id)
    print("Pass {0},Loss {1} NO.:{2}".format(pass_id,loss[0],x_.size))

plt.ioff()
plt.show()
