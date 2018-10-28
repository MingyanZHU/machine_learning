import numpy as np
import random
from scipy.stats import multivariate_normal
import collections


class GaussianMixtureModel(object):
    """ 高斯混合聚类EM算法 """
    def __init__(self, data, k=3, delta=1e-6, max_iteration=1000):
        self.data = data
        self.k = k
        self.delta = delta
        self.max_iteration = max_iteration
        self.__data_rows = len(self.data)
        self.__alpha = np.ones(self.k) * (1.0 / self.k)
        self.__mu, self.__sigma = self.__init_params()
        self.sample_assignments = None
        self.responsibilities = []
        self.responsibility = None

    def __init_params(self):
        mu = np.array(self.data[random.sample(range(self.__data_rows), self.k)])
        sigma = collections.defaultdict(list)
        for i in range(self.k):
            sigma[i] = np.cov(mu[i], mu[i]).tolist()
        # sigma = np.cov(mu)
        return mu, sigma

    def __likehoods(self):
        likehoods = np.zeros((self.__data_rows, self.k))
        for i in range(self.k):
            temp = multivariate_normal.pdf(self.data, self.__mu[i], self.__sigma[i])
            likehoods[:,i] = temp
        return likehoods
    
    def __expectation(self):
        weighted_likehoods = self.__likehoods() * self.__alpha
        sum_likehoods = np.expand_dims(np.sum(weighted_likehoods, axis=1), axis=1)
        self.responsibility = weighted_likehoods / sum_likehoods
        self.sample_assignments = self.responsibility.argmax(axis=1)
        self.responsibilities.append(np.max(self.responsibility, axis=1))

    ## 以下有问题
    def __maximization(self):
        for i in range(self.k):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            mean = (resp * self.data).sum(axis=0) / resp.sum()
            covariance = (self.data - mean).T.dot((self.data - mean) * resp) / resp.sum()
            self.__mu[i], self.__sigma[i] = mean, covariance
        self.__alpha = self.responsibility.sum(axis=0) / self.__data_rows

    def __converged(self):
        if len(self.responsibilities) < 2:
            return False
        diff = np.linalg.norm(
            self.responsibilities[-1] - self.responsibilities[-2])
        return diff <= self.delta

    def predict(self):
        for _ in range(self.max_iteration):
            self.__expectation()
            self.__maximization()
            if self.__converged():
                break
        self.__expectation()
        return self.sample_assignments
    # def _expectation(self):

# def calculate_covariance_matrix(X, Y=None):
#     """ Calculate the covariance matrix for the dataset X """
#     if Y is None:
#         Y = X
#     n_samples = np.shape(X)[0]
#     covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

#     return np.array(covariance_matrix, dtype=float)