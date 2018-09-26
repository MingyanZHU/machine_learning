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
    
    def fitting_standford(self, w_0):
        w = w_0
        r = self.b
        rhos = [np.linalg.norm(r) ** 2]
        for i in range(100000):
            if i == 0:
                p = r
            else:
                p = r + (rhos[i-1] / rhos[i-2]) * p
            omega = self.A @ p
            alpha = rhos[i-1] / (p.T @ omega)
            w = w + alpha * p
            r = r - alpha * omega
            rhos.append(np.linalg.norm(r) ** 2)
        return w

    def fitting_standford2(self, w_0):
        w = w_0
        r = self.b -self.A @ w_0
        M = np.linalg.inv(self.A)
        p = r
        z = M @ r
        rho_0 = r.T @ z
        for i in range(len(w_0)):
            omega = self.A @ p
            alpha = rho_0 / (omega @ p)
            w = w + alpha * p
            print(i, w)
            r = r - alpha * omega
            z = M @ r
            rho = z.T @ r
            p = z + rho / rho_0 * p
            rho_0 = rho
        return w
            