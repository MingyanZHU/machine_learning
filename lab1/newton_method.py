import numpy as np


class NewtonMethond(object):
    def __init__(self, X, T, hyper, delta=1e-6):
        """ 
        Args:
            X, T训练集, 其中X为(number_train, degree + 1)的矩阵
            T为(number_train, 1)的向量
            w_0:为初始化的解
        """
        self.X = X
        self.T = T
        self.hyper = hyper
        self.delta = delta

    def derivative(self, w):
        return np.transpose(self.X) @ self.X @ w + self.hyper * w - self.X.T @ self.T

    def second_derivative(self):
        return np.linalg.pinv(self.X.T @ self.X + self.hyper * np.identity(len(self.X.T[0])))

    def fitting(self, w_0):
        w = w_0
        while True:
            gradient = self.derivative(w)
            if np.linalg.norm(gradient) < self.delta:
                break
            wt = w - self.second_derivative() @ gradient
            w = wt
        return w
