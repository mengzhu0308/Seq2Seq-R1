#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/11/29 5:16
@File:          model.py
'''

import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.layers import *

from MyBidirectional import MyBidirectional
from MyLayerNormalization import MyLayerNormalization
from MyConditionalLayerNormalization import MyConditionalLayerNormalization
from MultiHeadAttention import MultiHeadAttention
from Loss import Seq2SeqLoss

'''搭建Seq2Seq模型，用于语言翻译任务
'''
def seq2seq_model(src_max_len, src_vocab_size, src_emsize, obj_max_len, obj_vocab_size, obj_emsize):
    x_in = Input(shape=(src_max_len, ))
    y_in = Input(shape=(obj_max_len, ))
    x_mask = Lambda(lambda arg: K.not_equal(arg, 0))(x_in)
    y_mask = Lambda(lambda arg: K.not_equal(arg, 0))(y_in)

    x_embedding = Embedding(src_vocab_size, src_emsize)
    y_embedding = Embedding(obj_vocab_size, obj_emsize)
    x, y = x_embedding(x_in), y_embedding(y_in)

    '''encoder，四层双向LSTM
    '''
    inter_emsize = src_emsize // 2
    x = MyLayerNormalization()(x)
    for _ in range(4):
        x = MyBidirectional(LSTM(inter_emsize, return_sequences=True))([x, x_mask])
        x = MyLayerNormalization()(x)
    x_max = Lambda(lambda inputs: K.max(inputs[0] - (1 - K.cast(inputs[-1], K.dtype(inputs[0]))[..., None]) * 1e10,
                                        axis=1))([x, x_mask])

    '''decoder，四层单向LSTM
    '''
    inter_emsize = max(src_emsize, obj_emsize) // 4
    y = MyConditionalLayerNormalization(hidden_units=inter_emsize)([y, x_max])
    for _ in range(4):
        y = LSTM(obj_emsize, return_sequences=True)(y)
        y = MyConditionalLayerNormalization(hidden_units=inter_emsize)([y, x_max])

    '''Attention交互
    '''
    xy = MultiHeadAttention(8, obj_emsize // 8)([y, x, x, x_mask])
    xy = concatenate([y, xy])

    '''分类
    '''
    xy = Dense(obj_emsize)(xy)
    xy = Activation(tf.nn.swish)(xy)
    xy = Dense(obj_vocab_size)(xy)
    xy = Softmax()(xy)
    xy = Seq2SeqLoss(output_axis=-1)([y_mask, y_in, xy])

    return Model([x_in, y_in], xy)