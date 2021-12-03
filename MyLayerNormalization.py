#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/29 18:59
@File:          MyLayerNormalization.py
'''

from keras import backend as K
from keras.layers import Layer

class MyLayerNormalization(Layer):
    def __init__(self, axis=1, center=True, scale=True, epsilon=None, **kwargs):
        super(MyLayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis if isinstance(axis, (list, tuple)) else (axis, )
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(MyLayerNormalization, self).build(input_shape)
        shape = [j if i in self.axis else 1 for i, j in enumerate(input_shape) if i != 0]
        self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta', trainable=self.center)
        self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma', trainable=self.scale)

    def call(self, inputs, **kwargs):
        ndim = len(K.int_shape(inputs))
        reduction_axes = [i for i in range(1, ndim) if i not in self.axis]

        outputs = inputs
        mean = K.mean(outputs, axis=reduction_axes, keepdims=True)
        outputs = outputs - mean
        variance = K.mean(K.square(outputs), axis=reduction_axes, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = outputs / std
        outputs = outputs * self.gamma[None, ...] + self.beta[None, ...]

        return outputs

    def compute_output_shape(self, input_shape):
            return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon
        }
        base_config = super(MyLayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))