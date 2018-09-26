import numpy as np


class NewtonMethond(object):
    def __init__(self, X, T, hyper, delta=1e-6):
        """ 
        Args:
            X, T训练集, 其中X为(number_train, degree + 1)的矩阵
            T为(number_train, 1)的向量
            hyper 为超参数
            delta 为迭代停止条件
        """
        self.X = X
        self.T = T
        self.hyper = hyper
        self.delta = delta

    def __derivative(self, w):
        """ 
        求函数的一阶导数 即 $J(w) = (X'X + \lambda I)w - X'T$
        """
        return np.transpose(self.X) @ self.X @ w + self.hyper * w \
        - self.X.T @ self.T

    def __second_derivative(self):
        """ 
        求函数的二阶导数的逆 即hessian矩阵的逆 """
        return np.linalg.pinv(self.X.T @ self.X + self.hyper \
            * np.identity(len(self.X.T[0])))

    def fitting(self, w_0):
        """ 
        用于牛顿法求解方程的解 """
        w = w_0
        while True:
            gradient = self.__derivative(w) # 梯度
            if np.linalg.norm(gradient) < self.delta:
                break
            wt = w - self.__second_derivative() @ gradient
            w = wt
        return w
