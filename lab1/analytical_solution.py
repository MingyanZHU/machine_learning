import numpy as np


class AnalyticalSolution(object):
    def __init__(self, X, T):
        self.X = X
        self.T = T

    def fitting(self):
        return np.linalg.pinv(self.X) @ self.T

    def fitting_with_regulation(self, hyper):
        return np.linalg.solve(np.identity(len(self.X.T)) * hyper + self.X.T @ self.X, self.X.T @ self.T)
    
    def E_rms(self, x, y):
        return np.sqrt(np.mean(np.square(x-y)))