#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/2/1 14:14
@File:          DataGenerator.py
'''

import numpy as np

from sampler import BatchSampler

class BaseDataGenerator:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        super(BaseDataGenerator, self).__init__()
        self.dataset = dataset
        self.index_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self._sampler_iter = iter(self.index_sampler)

    @property
    def sampler_iter(self):
        return self._sampler_iter

    def __len__(self):
        return len(self.index_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_data()

    def _next_index(self):
        try:
            index = next(self._sampler_iter)
        except StopIteration:
            self._sampler_iter = iter(self.index_sampler)
            index = next(self._sampler_iter)

        return index

    def _next_data(self):
        raise NotImplementedError

class DataGenerator(BaseDataGenerator):
    def __init__(self, dataset, **kwargs):
        super(DataGenerator, self).__init__(dataset, **kwargs)

    def _next_data(self):
        index = self._next_index()
        batch_x, batch_y = [], []
        for idx in index:
            x, y = self.dataset[idx]
            batch_x.append(x)
            batch_y.append(y)

        batch_x, batch_y = np.array(batch_x), np.array(batch_y)

        return [batch_x, batch_y], None