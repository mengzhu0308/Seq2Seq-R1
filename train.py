#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/11/29 5:16
@File:          train.py
'''

import math
import numpy as np
import pickle
from keras.optimizers import Nadam
from keras.callbacks import Callback

from model import seq2seq_model
from Dataset import Dataset
from DataGenerator import DataGenerator
from snippets import sent2ids, sequence_padding, ids2sent

if __name__ == '__main__':
    with open('eng_lang.pkl', 'rb') as f:
        eng_lang = pickle.load(f)
    with open('eng_df.pkl', 'rb') as f:
        eng_df = pickle.load(f)
    with open('chn_lang.pkl', 'rb') as f:
        chn_lang = pickle.load(f)
    with open('chn_df.pkl', 'rb') as f:
        chn_df = pickle.load(f)

    eng_min_freq = 2
    eng_id2token = ['<PAD>', 'UNK', 'START', 'END'] + [i for i, j in eng_df.items() if j >= eng_min_freq]
    eng_token2id = {j: i for i, j in enumerate(eng_id2token)}
    eng_vocab_size = len(eng_id2token)

    chn_min_freq = 4
    chn_id2token = ['<填充>', '<未知>', '<开始>', '<结束>'] + [i for i, j in chn_df.items() if j >= chn_min_freq]
    chn_token2id = {j: i for i, j in enumerate(chn_id2token)}
    chn_vocab_size = len(chn_id2token)
    
    topk = 3
    max_len = 64
    eng_emsize = 128
    chn_emsize = 128
    train_batch_size = 256
    epochs = 150
    init_lr = 0.001

    eng_lang = [sent2ids(sent, eng_token2id) for sent in eng_lang]
    chn_lang = [sent2ids(sent, chn_token2id, START_END=True) for sent in chn_lang]
    eng_lang = sequence_padding(eng_lang, max_seq_len=max_len)
    chn_lang = sequence_padding(chn_lang, max_seq_len=max_len + 2)

    train_dataset = Dataset(eng_lang, chn_lang)
    train_generator = DataGenerator(train_dataset, batch_size=train_batch_size, shuffle=True)

    model = seq2seq_model(max_len, eng_vocab_size, eng_emsize, max_len + 2, chn_vocab_size, chn_emsize)

    opt = Nadam(learning_rate=init_lr)
    model.compile(opt)

    '''beam search解码
       每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    '''
    def evaluate(s, topk=3, max_len=64):
        xids = np.array([sent2ids(s, eng_token2id)] * topk)  # 输入转id
        xids_padding = sequence_padding(xids, max_seq_len=max_len)
        yids = np.array([[2]] * topk)                        # 解码均以<开始>开头，这里<开始>的id为2
        yids_padding = sequence_padding(yids, max_seq_len=max_len + 2)
        scores = [0] * topk                                 # 候选答案分数
        for i in range(max_len):                            # 强制要求输出不超过max_len字
            proba = model.predict([xids_padding, yids_padding])[:, i, 3:]     # 直接忽略<填充>、<未知>、<开始>
            log_proba = np.log(proba + 1e-6)                 # 取对数，将累乘概率变成累加概率
            arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _yids = []      # 暂存的候选目标序列
            _scores = []    # 暂存的候选目标序列得分
            if i == 0:
                for j in range(topk):
                    _yids.append(list(yids[j]) + [arg_topk[0, j] + 3])
                    _scores.append(scores[j] + log_proba[0, arg_topk[0, j]])
            else:
                for j in range(topk):
                    for k in range(topk):                   # 遍历topk * topk的组合
                        _yids.append(list(yids[j]) + [arg_topk[j, k] + 3])
                        _scores.append(scores[j] + log_proba[j, arg_topk[j, k]])
                _arg_topk = np.argsort(_scores)[-topk:]     # 从中选出新的topk
                _yids = [_yids[k] for k in _arg_topk]
                _scores = [_scores[k] for k in _arg_topk]
            yids = np.array(_yids)
            scores = np.array(_scores)
            best_one = np.argmax(scores)
            if yids[best_one][-1] == 3:
                return ids2sent(yids[best_one], chn_id2token)
            yids_padding = sequence_padding(yids, max_seq_len=max_len + 2)

        # 如果max_len字都找不到<结束>，直接返回
        return ids2sent(yids[np.argmax(scores)], chn_id2token)

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()
            self.min_loss = math.inf

        def on_epoch_end(self, epoch, logs=None):
            print(evaluate('Hello, My name is Zhu Meng.', topk=topk, max_len=max_len))
            print(evaluate('It is interesting!', topk=topk, max_len=max_len))

            if logs['loss'] <= self.min_loss:
                self.min_loss = logs['loss']
                model.save_weights('seq2seq.weights')

    evaluator = Evaluator()

    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator],
                        shuffle=False,
                        initial_epoch=0)