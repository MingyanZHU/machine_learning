import numpy as np


class GradientDescentWithRegulation(object):
    def __init__(self, x, y, beta_0, hyper, rate=0.1, delta=1e-6):
        self.x = x
        self.y = y
        self.beta_0 = beta_0
        self.hyper = hyper
        self.rate = rate
        self.delta = delta
        self.__m = len(x)
        self.__n = len(x[0])  # 没有判断x是否为空

    def sigmod(self, z):
        return 1.0 / (1.0 + np.exp(z))

    def __loss(self, beta_t):
        ans = 0
        for i in range(self.__m):
            ans += (-self.y[i] * beta_t @ self.x[i] + np.log(1 +
                    np.exp(beta_t @ self.x[i])) + 0.5 * self.hyper * beta_t @ beta_t)
        return ans

    def __derivative_beta(self, beta_t):
        ans = np.zeros(self.__n)
        for i in range(self.__m):
            ans += (self.x[i] * (self.y[i] -
                     (1.0 - self.sigmod(beta_t @ self.x[i]))))
        return -1 * ans + self.hyper * beta_t

    def gradient(self):
        loss0 = self.__loss(self.beta_0)
        k = 0
        beta = self.beta_0
        while True:
            beta_t = beta - self.rate * self.__derivative_beta(beta)
            loss = self.__loss(beta_t)
            if np.abs(loss - loss0) < self.delta:
                break
            else:
                k = k + 1
                print(k)
                print("loss:", loss)
                # if loss > loss0:
                #     self.rate *= 0.5
                # 进行学习率的衰减 得到的结果不正确??
                loss0 = loss
                beta = beta_t
        return beta

    def predict(self, beta):
        ans = np.zeros(self.__m)
        for i in range(self.__m):
            if self.sigmod(beta @ self.x[i]) < 0.5:
                ans[i] = 1
        return ans
