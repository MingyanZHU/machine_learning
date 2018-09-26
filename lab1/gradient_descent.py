import numpy as np


class GradientDescent(object):
    def __init__(self, X, T, hyper, rate=0.1, delta=1e-6):
        """ Args:
            X, T训练集, 其中X为(number_train, degree + 1)的矩阵
            T为(number_train, 1)的向量
            hyper 为超参数
            rate 为学习率, delta 为停止迭代的条件
        """
        self.X = X
        self.T = T
        self.hyper = hyper
        self.rate = rate
        self.delta = delta

    def __loss(self, w):
        """
        用于求解loss 在本次实验中loss的公式为
        $E(w) = \frac{1}{2N}[(Xw - T)'(Xw - T) + \lambda w'w]$ 
        """
        wt = np.transpose([w])
        temp = self.X @ w - self.T
        return 0.5 * np.mean(temp.T @ temp + self.hyper * w @ wt)

    def __derivative(self, w):
        """ 
        一阶函数求导 """
        return np.transpose(self.X) @ self.X @ w + self.hyper * w -\
         self.X.T @ self.T

    def fitting(self, w_0):
        """ 
        用于多项式函数使用梯度下降拟合
        Args:
            w_0 初始解，通常以全零向量
         """
        loss0 = self.__loss(w_0)
        k = 0
        w = w_0
        while True:
            wt = w - self.rate * self.__derivative(w)
            loss = self.__loss(wt)
            if np.abs(loss - loss0) < self.delta:
                break
            else:
                k = k + 1
                # print(k) # 记录迭代次数
                # print("loss:", loss)
                if loss > loss0:
                    # 当loss函数相比上次增大 学习率衰减为之前的一半
                    self.rate *= 0.5
                loss0 = loss
                w = wt
        return w
