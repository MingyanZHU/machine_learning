import numpy as np


class ConjugateGradient(object):
    def __init__(self, X, T, hyper, delta=1e-6):
        self.X = X
        self.T = T
        self.hyper = hyper
        self.delta = delta
        self.A = X.T @ X + np.identity(len(X.T)) * hyper
        self.b = X.T @ T

    def fitting(self, w_0):
        r_0 = self.b - self.A @ w_0
        w = w_0
        p = r_0
        k = 0
        while True:
            k = k + 1
            alpha = np.linalg.norm(r_0) ** 2 / (np.transpose(p) @ self.A @ p)
            w = w + alpha * p
            r = r_0 - alpha * self.A @ p
            print(k, r)
            if(np.linalg.norm(r) ** 2 < self.delta):
                break
            beta = np.linalg.norm(r)**2 / np.linalg.norm(r_0)**2
            p = r + beta * p
            r_0 = r
        return w