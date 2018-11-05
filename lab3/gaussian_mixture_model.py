import numpy as np
import random
from scipy.stats import multivariate_normal
import collections


class GaussianMixtureModel(object):
    """ 高斯混合聚类EM算法 """

    def __init__(self, data, k=3, delta=1e-12, max_iteration=1000):
        self.data = data
        self.k = k
        self.delta = delta
        self.max_iteration = max_iteration
        self.data_rows, self.data_columns = self.data.shape
        self.__alpha = np.ones(self.k) * (1.0 / self.k)
        self.__mu, self.__sigma = self.__init_params()
        self.sample_assignments = None
        self.c = collections.defaultdict(list)
        self.__last_alpha = self.__alpha
        self.__last_mu = self.__mu
        self.__last_sigma = self.__sigma
        self.__gamma = None

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
            for i in range(self.data_rows):
                temp_ans.append(np.sum([self.__euclidean_distance(
                    self.data[i], mu[j]) for j in range(len(mu))]))
            mu.append(self.data[np.argmax(temp_ans)])
        return np.array(mu)

    def __init_params(self):
        # mu = np.array(self.data[random.sample(range(self.data_rows), self.k)])
        # 随机选择k个点作为初始点 极易陷入局部最小值
        mu = self.__initial_center_not_random()
        sigma = collections.defaultdict(list)
        for i in range(self.k):
            sigma[i] = np.eye(self.data_columns, dtype=float) * 0.1
        return mu, sigma

    # def __gaussian(self, mean, cov):
    #     det = np.linalg.det(cov)
    #     cov_i = np.linalg.pinv(cov)
    #     temp_x = np.math.pow(2 * np.pi, 0.5 * self.data_columns) * np.math.pow(det, 0.5)
    #     temp_y = np.exp(-0.5 * (self.data[5] - mean).T.dot(cov_i).dot(self.data[5] - mean))
    #     return 1.0 * temp_y / temp_x

    def __likelihoods(self):
        likelihoods = np.zeros((self.data_rows, self.k))
        for i in range(self.k):
            likelihoods[:, i] = multivariate_normal.pdf(self.data, self.__mu[i], self.__sigma[i])
        return likelihoods

    def __expectation(self):
        # 求期望 E
        weighted_likelihoods = self.__likelihoods() * self.__alpha    # (m,k)
        sum_likelihoods = np.expand_dims(np.sum(weighted_likelihoods, axis=1), axis=1)  # (m,1)
        print(np.log(np.prod(sum_likelihoods)))     # 输出似然值
        self.__gamma = weighted_likelihoods / sum_likelihoods    # (m,k)
        self.sample_assignments = self.__gamma.argmax(axis=1)    # (m,)
        for i in range(self.data_rows):
            self.c[self.sample_assignments[i]].append(self.data[i].tolist())

    def __maximization(self):
        # 最大化 M
        for i in range(self.k):
            gamma = np.expand_dims(self.__gamma[:, i], axis=1)    # 提取每一列 作为列向量 (m, 1)
            mean = (gamma * self.data).sum(axis=0) / gamma.sum()
            covariance = (self.data - mean).T.dot((self.data - mean) * gamma) / gamma.sum()
            self.__mu[i], self.__sigma[i] = mean, covariance    # 更新参数
        self.__alpha = self.__gamma.sum(axis=0) / self.data_rows

    def __converged(self):
        # 迭代终止条件 参数sigma mu和alpha几乎不变化
        diff = np.linalg.norm(self.__last_alpha - self.__alpha) \
               + np.linalg.norm(self.__last_mu - self.__mu) \
               + np.sum([np.linalg.norm(self.__last_sigma[i] - self.__sigma[i]) for i in range(self.k)])
        if diff > self.delta:
            self.__last_sigma = self.__sigma
            self.__last_mu = self.__mu
            self.__last_alpha = self.__alpha
            return False
        else:
            return True

    def predict(self):
        print("GMM")
        for i in range(self.max_iteration):
            print(i)
            self.__expectation()
            self.__maximization()
            if self.__converged():
                break
        self.__expectation()
        return self.__mu, self.c
