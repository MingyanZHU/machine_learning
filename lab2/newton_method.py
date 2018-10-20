import numpy as np

""" 牛顿法 """
class NewtonMethod(object):
    def __init__(self, x, y, beta_0, hyper=0, delta=1e-6):
        self.x = x
        self.y = y
        self.beta_0 = beta_0
        self.hyper = hyper
        self.delta = delta
        self.__m = len(x)
        self.__n = len(x[0])  # 没有判断x是否为空

    def __sigmod(self, z):
        return 1.0 / (1.0 + np.exp(z))

    def __derivative(self, beta_t):
        ans = np.zeros(self.__n)
        for i in range(self.__m):
            ans += (self.x[i] * (self.y[i] -
                    (1.0 - self.__sigmod(beta_t @ self.x[i]))))
        return -1 * ans + self.hyper * beta_t

    def __second_derivative(self, beta_t):
        """ 求二阶导黑塞矩阵的逆 """
        ans = np.eye(self.__n) * self.hyper
        for i in range(self.__m):
            temp = self.__sigmod(beta_t @ self.x[i])
            ans += self.x[i] * np.transpose([self.x[i]]) * temp * (1 - temp)
        return np.linalg.pinv(ans)

    def fitting(self):
        beta = self.beta_0
        while True:
            gradient = self.__derivative(beta)
            if np.linalg.norm(gradient) < self.delta:
                break
            beta_t = beta - self.__second_derivative(beta) @ gradient
            beta = beta_t
        return beta
