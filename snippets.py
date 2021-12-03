#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/2/2 10:20
@File:          snippets.py
'''

import numpy as np

def sent2ids(sent, token2id, UNK=1, START_END=False):
    target_sent = [2] if START_END else []
    for w in sent:
        try:
            target_sent.append(token2id[w])
        except KeyError:
            target_sent.append(UNK)
    if START_END:
        target_sent.append(3)

    return np.array(target_sent, dtype='int32')

def ids2sent(ids, id2token):
    s = ''.join([id2token[i] for i in ids if i not in (0, 1, 2, 3)])
    return s

'''Numpy函数，将序列padding到同一长度
'''
def sequence_padding(inputs, max_seq_len=None, padding=0):
    if max_seq_len is None:
        max_seq_len = max([len(x) for x in inputs])

    outputs = []
    for x in inputs:
        x = x[:max_seq_len]
        pad_width = (0, max_seq_len - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)