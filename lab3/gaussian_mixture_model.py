import math
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
        self.data_rows, self.data_columns = self.data.shape
        self.alpha = np.ones(self.k) * (1.0 / self.k)
        self.mu, self.sigma = self.init_params()
        self.sample_assignments = None
        self.responsibilities = []
        self.responsibility = None

    def init_params(self):
        mu = np.array(self.data[random.sample(range(self.data_rows), self.k)])
        sigma = collections.defaultdict(list)
        for i in range(self.k):
            sigma[i] = np.eye(self.data_columns, dtype=float) * 0.1
        return mu, sigma

    def gaussian(self, index):
        mean = self.mu[index]
        cov = self.sigma[index]
        det = np.linalg.det(cov)
        likehoods = np.zeros(self.data_rows)
        for i, sample in enumerate(self.data):
            numerator = np.power(2 * np.pi, self.data_columns / 2) * np.math.sqrt(det)
            order = np.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(cov)).dot((sample - mean)))
            likehoods[i] = (1.0 / numerator) * order
        
        return likehoods
    
    def multivariate_gaussian(self, X, index):
        n_features = np.shape(X)[1]
        mean = self.mu[index]
        covar = self.sigma[index]
        determinant = np.linalg.det(covar)
        likelihoods = np.zeros(np.shape(X)[0])
        for i, sample in enumerate(X):
            d = n_features
            coeff = (1. / (math.pow(2 * np.pi, d / 2) * math.sqrt(determinant)))
            exponent = np.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods
    
    def scipy_gaussian(self, index):
        return multivariate_normal.pdf(self.data, self.mu[index], self.sigma[index])

    def llikehoods(self):
        likehoods = np.zeros((self.data_rows, self.k))
        for i in range(self.k):
            temp = multivariate_normal.pdf(self.data, self.mu[i], self.sigma[i])
            likehoods[:,i] = temp
        return likehoods
    
    def expectation(self):
        weighted_likehoods = self.llikehoods() * self.alpha
        sum_likehoods = np.expand_dims(np.sum(weighted_likehoods, axis=1), axis=1)
        self.responsibility = weighted_likehoods / sum_likehoods
        self.sample_assignments = self.responsibility.argmax(axis=1)
        self.responsibilities.append(np.max(self.responsibility, axis=1))

    ## 以下有问题
    def maximization(self):
        for i in range(self.k):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            mean = (resp * self.data).sum(axis=0) / resp.sum()
            covariance = (self.data - mean).T.dot((self.data - mean) * resp) / resp.sum()
            self.mu[i], self.sigma[i] = mean, covariance
        self.alpha = self.responsibility.sum(axis=0) / self.data_rows

    def converged(self):
        if len(self.responsibilities) < 2:
            return False
        diff = np.linalg.norm(
            self.responsibilities[-1] - self.responsibilities[-2])
        return diff <= self.delta

    def predict(self):
        for _ in range(self.max_iteration):
            self.expectation()
            self.maximization()
            if self.converged():
                break
        self.expectation()
        return self.sample_assignments

""" def watermelon_data():
    watermelon = np.array([[0.697, 0.46],
                           [0.774, 0.376],
                           [0.634, 0.264],
                           [0.608, 0.318],
                           [0.556, 0.215],
                           [0.403, 0.237],
                           [0.481, 0.149],
                           [0.437, 0.211],
                           [0.666, 0.091],
                           [0.243, 0.267],
                           [0.245, 0.057],
                           [0.343, 0.099],
                           [0.639, 0.161],
                           [0.657, 0.198],
                           [0.36, 0.37],
                           [0.593, 0.042],
                           [0.719, 0.103],
                           [0.359, 0.188],
                           [0.339, 0.241],
                           [0.282, 0.257],
                           [0.748, 0.232],
                           [0.714, 0.346],
                           [0.483, 0.312],
                           [0.478, 0.437],
                           [0.525, 0.369],
                           [0.751, 0.489],
                           [0.532, 0.472],
                           [0.473, 0.376],
                           [0.725, 0.445],
                           [0.446, 0.459]])
    return watermelon

def generate_data(sample_means, sample_number, number_k):
    """ 生成2维数据
    :argument number_k k类
    :argument sample_means k类数据的均值 以list的形式给出 如[[1, 2],[-1, -2], [0, 0]]
    :argument sample_number k类数据的数量 以list的形式给出 如[10, 20, 30]
    """
    assert number_k > 0
    assert len(sample_means) == number_k
    assert len(sample_number) == number_k
    cov = [[0.1, 0], [0, 0.1]]
    sample_data = []
    for index in range(number_k):
        for times in range(sample_number[index]):
            sample_data.append(np.random.multivariate_normal(
                [sample_means[index][0], sample_means[index][1]], cov).tolist())
    return np.array(sample_data)


def main():
    k = 3
    means = [[2, 4], [0, -4], [-2, 2]]
    number = [100, 100, 100]
    data = generate_data(means, number, k)
    # data = watermelon_data()
    print(data.shape)
    gmm = GaussianMixtureModel(data)
    # mu, sigma = gmm.init_params()
    # print(mu)
    # print(sigma)
    # likehoods = gmm.gaussian(2)
    # print(likehoods)
    # likehoods = gmm.multivariate_gaussian(data, 2)
    # print(likehoods)
    # print(gmm.scipy_gaussian(1).shape)
    print(gmm.predict())

if __name__ =='__main__':
    main()


""" # def calculate_covariance_matrix(X, Y=None):
#     """ Calculate the covariance matrix for the dataset X """
#     if Y is None:
#         Y = X
#     n_samples = np.shape(X)[0]
#     covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

#     return np.array(covariance_matrix, dtype=float) """ """