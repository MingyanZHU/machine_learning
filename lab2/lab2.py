import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom

import gradient_descent

""" TODO
    1.首先不生成数据，利用已有的数据(如西瓜书上的例子)先进行测试
    2.尽量使用lab1中已经造好的轮子进行
 """


def loadWaterMelonData():
    dataSet = np.loadtxt("./watermelon-3.3.csv", delimiter=",")
    x = dataSet[:, 1:3]
    y = dataSet[:, 3]
    return x, y

x, y = loadWaterMelonData()
m, n = np.shape(x)
x = np.c_[np.ones(m), x]

gd = gradient_descent.GradientDescent(x, y, np.zeros(n+1))
ans = gd.gradient()
print(gd.predict(ans))

x_draw = np.linspace(0, 1)
y_draw = - (ans[0] + ans[1] * x_draw) / ans[2]
type1_x = []
type1_y = []
type2_x = []
type2_y = []
for i in range(len(x)):
    if y[i] == 0:
        type1_x.append(x[i][1])
        type1_y.append(x[i][2])
    else:
        type2_x.append(x[i][1])
        type2_y.append(x[i][2])

plt.scatter(type1_x, type1_y, facecolor="none", edgecolor="b", label="positive")
plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")
plt.plot(x_draw, y_draw)
plt.legend()
plt.show()

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

# # k = np.arange(0, number + 1)
# # binomial = binom.pmf(k, number, p)
# # print(binomial)
