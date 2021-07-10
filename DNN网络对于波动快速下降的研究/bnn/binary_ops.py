# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    使用全梯度传播将元素舍入到最接近的整数。
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)

    tf.keras.backend.round(x):以元素方式四舍五入到最接近的整数。
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation

    用于训练二值化神经网络的二进制硬S形。
      神经元的激活二值化功能
      它在正向传播期间的行为类似于符号函数
      就像：
         hard_tanh（x）= 2 * _hard_sigmoid（x）-1
         | x |时清除梯度 反向传播期间> 1

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1):
    '''The weights' binarization function, 

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    # [-H, H] -> -H or H
    Wb = H * binary_tanh(W / H)
    return Wb


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb
