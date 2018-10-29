import numpy as np
import pandas as pd


class IrisProcessing(object):
    def __init__(self):
        self.data_set = pd.read_csv("./iris.csv")
        self.data_set['class'] = self.data_set['class'].map(
            {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}).astype(int)
        self.x = self.data_set.drop('class', axis=1)
        self.y = self.data_set['class']

    def get_data(self):
        return np.array(self.x, dtype=float)

    def accuracy(self, y_label):
        count = 0
        number = len(self.y)
        for i in range(number):
            if self.y[i] == y_label[i]:
                count = count + 1
        return count * 1.0 / number
