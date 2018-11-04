import numpy as np
import pandas as pd
import itertools

class IrisProcessing(object):
    def __init__(self):
        self.data_set = pd.read_csv("./iris.csv")
        # self.data_set['class'] = self.data_set['class'].map(
        #     {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}).astype(int)
        self.x = self.data_set.drop('class', axis=1)
        self.y = self.data_set['class']
        self.classes = list(itertools.permutations(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 3))

    def get_data(self):
        return np.array(self.x, dtype=float)

    def acc(self, y_label):
        """ 用于测试聚类的正确率 """
        number = len(self.y)
        counts = []
        for i in range(len(self.classes)):
            count = 0
            for j in range(number):
                if self.y[j] == self.classes[i][y_label[j]]:
                    count += 1
            counts.append(count)
        return np.max(counts) * 1.0 / number