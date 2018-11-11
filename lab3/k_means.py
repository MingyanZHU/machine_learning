import numpy as np
import collections
import random


class KMeans(object):
    def __init__(self, data, k, delta=1e-6):
        self.data = data
        self.k = k
        self.delta = delta
        self.__data_rows, self.__data_columns = data.shape
        self.__mu = self.__initial_center_not_random()
        self.sample_assignments = [-1] * self.__data_rows

    @staticmethod
    def __euclidean_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def __initial_center_not_random(self):
        """ 选择彼此距离尽可能远的K个点 """
        # 随机选第1个初始点
        mu_0 = np.random.randint(0, self.k) + 1
        mu = [self.data[mu_0]]
        # 依次选择与当前mu中样本点距离最大的点作为初始簇中心点
        for times in range(self.k-1):
            temp_ans = []
            for i in range(self.__data_rows):
                temp_ans.append(np.sum([self.__euclidean_distance(
                    self.data[i], mu[j]) for j in range(len(mu))]))
            mu.append(self.data[np.argmax(temp_ans)])
        return np.array(mu)

    def __k_means(self):
        times = 0
        while True:
            c = collections.defaultdict(list)
            for i in range(self.__data_rows):
                dij = [self.__euclidean_distance(
                    self.data[i], self.__mu[j]) for j in range(self.k)]
                lambda_j = np.argmin(dij)
                c[lambda_j].append(self.data[i].tolist())
                self.sample_assignments[i] = lambda_j

            new_mu = np.array([np.mean(c[i], axis=0).tolist()
                               for i in range(self.k)])

            loss = np.sum(self.__euclidean_distance(
                self.__mu[i], new_mu[i]) for i in range(self.k))
            if loss > self.delta:
                self.__mu = new_mu
            else:
                break

            print("K-means", times)
            times = times + 1
            print(self.__mu)
        return self.__mu, c

    def k_means_random_center(self):
        """ 随机选择k个顶点作为初始簇中心点 """
        self.__mu = self.data[random.sample(range(self.__data_rows), self.k)]
        return self.__k_means()

    def k_means_not_random_center(self):
        """ 随机选择第一个簇中心点 再选择彼此距离最大的k个顶点作为初始簇中心点 """
        self.__mu = self.__initial_center_not_random()
        return self.__k_means()
