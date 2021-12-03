#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/11/29 11:19
@File:          MultiHeadAttention.py
'''

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, Dense
from keras import initializers

'''多头注意力机制
'''
class MultiHeadAttention(Layer):
    def __init__(self, num_heads, head_size, key_size=None, use_bias=False, kernel_initializer='glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_dim = num_heads * head_size
        self.key_size = key_size if key_size else head_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.num_heads, use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.k_dense = Dense(self.key_size * self.num_heads, use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.v_dense = Dense(self.out_dim, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer)

    def mask(self, x, mask):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            mask = K.cast(mask, K.dtype(x))
            return x * mask - (1 - mask) * 1e12

    def call(self, inputs, **kwargs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.num_heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.num_heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.num_heads, self.head_size))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = tf.einsum('ijkl, ijml->ijkm', qw, kw) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask)
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('ijkl, ijlm->ijkm', a, vw)
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))