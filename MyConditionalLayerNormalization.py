#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/11/29 6:54
@File:          MyConditionalLayerNormalization.py
'''

from keras import backend as K
from keras.layers import Layer, Dense, Lambda
from keras import activations, initializers

class MyConditionalLayerNormalization(Layer):
    def __init__(self,
                 axis=1,
                 center=True,
                 scale=True,
                 epsilon=None,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform', **kwargs):
        super(MyConditionalLayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis if isinstance(axis, (list, tuple)) else (axis, )
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)

    def build(self, input_shape):
        super(MyConditionalLayerNormalization, self).build(input_shape)
        shape = [j if i in self.axis else 1 for i, j in enumerate(input_shape[0]) if i != 0]
        self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta', trainable=self.center)
        if self.center:
            self.beta_dense = Dense(input_shape[0][-1], use_bias=False, kernel_initializer='zeros')
        else:
            self.beta_dense = Lambda(lambda inputs: inputs)
        self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma', trainable=self.scale)
        if self.scale:
            self.gamma_dense = Dense(input_shape[0][-1], use_bias=False, kernel_initializer='zeros')
        else:
            self.gamma_dense = Lambda(lambda inputs: inputs)

        if self.hidden_units is not None:
            self.hidden_dense = Dense(self.hidden_units, activation=self.hidden_activation, use_bias=False,
                                      kernel_initializer=self.hidden_initializer)
        else:
            self.hidden_dense = Lambda(lambda inputs: inputs)

    def call(self, inputs, **kwargs):
        x, cond = inputs
        ndim = len(K.int_shape(x))
        reduction_axes = [i for i in range(1, ndim) if i not in self.axis]

        cond = self.hidden_dense(cond)
        for _ in range(K.ndim(x) - K.ndim(cond)):
            cond = K.expand_dims(cond, axis=1)

        mean = K.mean(x, axis=reduction_axes, keepdims=True)
        x = x - mean
        variance = K.mean(K.square(x), axis=reduction_axes, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = x / std
        x = x * (self.gamma[None, ...] + self.gamma_dense(cond)) + (self.beta[None, ...] + self.beta_dense(cond))

        return x

    def compute_output_shape(self, input_shape):
            return input_shape[0]

    def get_config(self):
        config = {
            'axis': self.axis,
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer': initializers.serialize(self.hidden_initializer)
        }
        base_config = super(MyConditionalLayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))