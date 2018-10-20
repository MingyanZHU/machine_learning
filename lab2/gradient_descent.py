import numpy as np

""" 梯度下降法 """
class GradientDescent(object):
    def __init__(self, x, y, beta_0, hyper=0, rate=0.1, delta=1e-6):
        self.x = x
        self.y = y
        self.beta_0 = beta_0
        self.hyper = hyper
        self.rate = rate
        self.delta = delta
        self.__m = len(x)
        self.__n = len(x[0])  # 没有判断x是否为空

    def __sigmod(self, z):
        return 1.0 / (1.0 + np.exp(z))

    def __loss(self, beta_t):
        ans = 0
        for i in range(self.__m):
            ans += (-self.y[i] * beta_t @ self.x[i] + np.log(1 + np.exp(beta_t @ self.x[i])))
        return (ans + 0.5 * self.hyper * beta_t @ beta_t) / self.__m
        # 此处m用于平衡loss 没有其他作用

    def __derivative_beta(self, beta_t):
        ans = np.zeros(self.__n)
        for i in range(self.__m):
            ans += (self.x[i] * (self.y[i] -
                                 (1.0 - self.__sigmod(beta_t @ self.x[i]))))
        return (-1 * ans + self.hyper * beta_t) / self.__m

    def fitting(self):
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
                if loss > loss0:
                    self.rate *= 0.5
                # 进行学习率的衰减 得到的结果不正确??
                # 修改后答案正确 原因可能为hyper取值过大
                loss0 = loss
                beta = beta_t
        return beta

    def predict(self, beta):
        # 好像没用的函数..
        ans = np.zeros(self.__m)
        for i in range(self.__m):
            if self.__sigmod(beta @ self.x[i]) < 0.5:
                ans[i] = 1
        return ans
