'''Trains a simple binarize fully connected NN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 97.9% test accuracy after 20 epochs using theano backend
''在MNIST数据集上训练简单的二值化完全连接的NN。
从keras的示例/mnist_mlp.py修改
使用theano后端20个纪元后达到97.9％的测试准确度
'''



from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import datetime

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import keras

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip

from keras.models import load_model


class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    Keras Dropout确实会在训练阶段扩展输入，这在这里是不希望的。
    '''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

def binary_tanh(x):
    return binary_tanh_op(x)




#batch_size = 4096
batch_size = 128
# 从之前的20000次训练先改为200/600次训练
epochs = 1
#epochs = 800
#epochs = 1
nb_classes = 10

#H = 'Glorot'
#kernel_lr_multiplier = 'Glorot'
H = 1
kernel_lr_multiplier = 10

# network
num_unit = 1024
num_hidden = 3
use_bias = True
stddev = 100
#bias_initializer = 'zeros'
bias_initializer = keras.initializers.RandomNormal(mean=0,stddev=stddev)

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
drop_in = 0.2
drop_hidden = 0.5


start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S");

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 神经网络的输入是28x28=784个结点用于接收图片的像素，因此需要先将每张图片从28x28转换成784的列表，因此使用了reshape函数
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 由于其数值范围从0到255，因此除以255以实现归一化使得数据范围从0到1
X_train /= 255
X_test /= 255
X_train[X_train>0]=1
X_train[X_train==0]=-1
X_test[X_test>0]=1
X_test[X_test==0]=-1
#print(X_train[0])
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 将类向量转换为二进制类矩阵
# 将标签转换成one hot（独热）格式，以便对应生成的十个输出神经元
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

model = Sequential()

#
model.add(DropoutNoScale(drop_in, input_shape=(784,), name='drop0'))
for i in range(num_hidden):
    model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
              name='dense{}'.format(i+1),bias_initializer=bias_initializer))
    #model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
    model.add(Activation(binary_tanh, name='act{}'.format(i+1)))
    model.add(DropoutNoScale(drop_hidden, name='drop{}'.format(i+1)))
model.add(BinaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
          name='dense',bias_initializer=bias_initializer))
#model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))
model.summary()

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])


# deserialized custom layers
#model.save('mlp.h5')
#model = load_model('mlp.h5', custom_objects={'DropoutNoScale': DropoutNoScale,
#                                             'BinaryDense': BinaryDense,
#                                             'Clip': Clip, 
#                                             'binary_tanh': binary_tanh})

print('batch%d_hidden%d_epoch%d_std%d'%(batch_size,num_unit,epochs,stddev)+start_time)
#############training process
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


np.save('batch%d_hidden%d_epoch%d_std%d'%(batch_size,num_unit,epochs,stddev)+start_time+'.npy',history.history)
#

#################model saving###############
model.save('batch%d_hidden%d_epoch%d_std%d'%(batch_size,num_unit,epochs,stddev)+start_time+'.h5')

#model = load_model('batch128_hidden128_epoch1000.h5', custom_objects={'DropoutNoScale': DropoutNoScale,
#                                             'BinaryDense': BinaryDense,
#                                             'Clip': Clip, 
#                                             'binary_tanh': binary_tanh})


############middle layer output#################
#inp = model.input                                           # input placeholder
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
#
## Testing 1
#test = np.random.random(X_test[0].shape)[np.newaxis,...]
##layer_outs = functor([X_test[0], 1.]) #with dropout
#layer_outs = functor([X_test[0][np.newaxis,...], 0.]) #without dropout
#print(layer_outs)

## Testing 1
#test = np.random.random(X_test.shape)
##layer_outs = functor([X_test[0], 1.]) #with dropout
#layer_outs = functor([X_test, 0.]) #without dropout
#print(layer_outs)
weight=model.get_weights()
np.save('weight_retrain15.npy', weight)