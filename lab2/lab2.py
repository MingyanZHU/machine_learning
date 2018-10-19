import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom

""" TODO
    1.首先不生成数据，利用已有的数据(如西瓜书上的例子)先进行测试
    2.尽量使用lab1中已经造好的轮子进行
 """
def loadWaterMelonData():
    dataSet =  np.loadtxt("./watermelon-3.3.csv",delimiter=",")
    x = dataSet[:,1:3]
    y = dataSet[:,3]
    return x, y

def loss(x, y, beta):
    sum = 0
    for i in range(len(beta)):
        sum += (-y[i] * beta @ x[i] + np.math.log(1 + np.math.exp(beta @ x[i])))
    return sum
def sigmod(x, beta):
    return 1.0 / (1.0 + np.math.exp(beta @ x))
def derivative_beta(x, y, beta):
    m,n = x.shape
    ans = np.zeros(n)
    for i in range(m):
        temp = y[i] - (1 - sigmod(x[i], beta))
        for j in range(n):
            ans[j] += x[i][j] * temp
    return -1 * ans  # TODO 看西瓜书3.30式 推导一下导数式向量还是标量
x, y = loadWaterMelonData()
print(x)
print(y)
m,n = np.shape(x)
x = np.c_[np.ones(m),x]
print(x)
beta = np.zeros(n+1)
print(beta)
print(derivative_beta(x, y, beta))
print(loss(x,y,beta))


# def generateData(number, meanPos, meanNeg, scale1=0.1, scale2=0.2):
#     y = np.random.randint(0, 2, (number, 1))
#     meanPos_1 = meanPos * np.random.random_sample()
#     meanPos_2 = meanPos * np.random.random_sample()
#     meanNeg_1 = meanNeg * np.random.random_sample()
#     meanNeg_2 = meanNeg * np.random.random_sample()
#     x_1 = []
#     x_2 = []
#     for i in range(number):
#         x_1.append(np.random.)
#     x_1 = np.random.normal(loc=mean_1, scale=scale1, size=(number, 1))
#     x_2 = np.random.normal(loc=mean_2, scale=scale2, size=(number, 1))
#     return x_1, x_2, y


# number = 20
# mean_1 = 2
# mean_2 = -2
# x_1, x_2, y = generateData(number, mean_1, mean_2)
# print(x_1.T)
# print(x_2.T)

# type1_x = []
# type1_y = []
# type2_x = []
# type2_y = []
# for i in range(number):
#     if y[i] == 0:
#         type1_x.append(x_1[i])
#         type1_y.append(x_2[i])
#     else:
#         type2_x.append(x_1[i])
#         type2_y.append(x_2[i])

# plt.scatter(type1_x, type1_y, facecolor="none", edgecolor="b", label="positive")
# plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")
# plt.legend()
# plt.show()

# # k = np.arange(0, number + 1)
# # binomial = binom.pmf(k, number, p)
# # print(binomial)
