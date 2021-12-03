#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/11/29 10:09
@File:          MyBidirectional.py
'''

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class MyBidirectional(Layer):
    def __init__(self, layer, **kwargs):
        super(MyBidirectional, self).__init__(**kwargs)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def _reverse_sequence(self, x, mask):
        seq_len = K.sum(K.cast(mask, 'int32'), axis=1)
        return tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        x_forward = self.forward_layer(x)
        x_backward = self._reverse_sequence(x, mask)
        x_backward = self.backward_layer(x_backward)
        x_backward = self._reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2,)