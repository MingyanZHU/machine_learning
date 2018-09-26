import numpy as np


class AnalyticalSolution(object):
    def __init__(self, X, T):
        """ 求方程的解析解 """
        self.X = X
        self.T = T

    def fitting(self):
        """ 不带惩罚项的解析解 """
        return np.linalg.pinv(self.X) @ self.T

    def fitting_with_regulation(self, hyper):
        """ 带惩罚项的解析解 """
        return np.linalg.solve(np.identity(len(self.X.T)) * hyper + \
        self.X.T @ self.X, self.X.T @ self.T)
    
    def E_rms(self, x, y):
        """ 根均方(RMS)误差 """
        return np.sqrt(np.mean(np.square(x-y)))