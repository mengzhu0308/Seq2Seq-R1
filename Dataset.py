'''
@Author:        ZM
@Date and Time: 2019/10/8 6:28
@File:          Dataset.py
'''

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        return x, y