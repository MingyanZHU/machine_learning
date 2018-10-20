import mushroom_read
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.stats import binom

import gradient_descent
import gradient_descent_with_regulation
import newton_methon


def generateData(number, meanPos, meanNeg, proportion_pos, scale1=0.1, scale2=0.2):
    """ 
    args:
        number:样本的数量
        meanPos:正例的平均值
        meanNeg:反例的平均值
        proportion_pos:正例占样本的数量
        scale1:正例的高斯分布的方差
        scale2:反例的高斯分布的方差
     """
    assert(proportion_pos > 0 and proportion_pos < 1)
    x = []
    y = []
    number_pos = int(number * proportion_pos)
    print("pos", number_pos)
    number_neg = number - number_pos
    meanPos_1 = meanPos * np.random.random_sample()
    meanPos_2 = meanPos * np.random.random_sample()
    meanNeg_1 = meanNeg * np.random.random_sample()
    meanNeg_2 = meanNeg * np.random.random_sample()
    for i in range(number_pos):
        x1 = random.gauss(meanPos_1, scale1)
        x2 = random.gauss(meanPos_2, scale1)
        x.append([x1, x2])
        y.append(1)
    for i in range(number_neg):
        x1 = random.gauss(meanNeg_1, scale2)
        x2 = random.gauss(meanNeg_2, scale2)
        x.append([x1, x2])
        y.append(0)
    return x, np.array(y)


def loadWaterMelonData():
    """ 读取西瓜书3.3a的数据 """
    dataSet = np.loadtxt("./watermelon-3.3.csv", delimiter=",")
    x = dataSet[:, 1:3]
    y = dataSet[:, 3]
    return x, y


# x, y = loadWaterMelonData()
# m, n = np.shape(x)
# x = np.c_[np.ones(m), x]
# print(y.shape)
# gdr = gradient_descent_with_regulation.GradientDescentWithRegulation(
#     x, y, np.zeros(n+1), 1e-4)
# nw = newton_methon.NewtonMethod(x, y, np.zeros(n + 1), 1e-4)

# # ans = nw.fitting()
# ans = gdr.gradient()
# print(ans)
# print(gdr.predict(ans))
# x_draw = np.linspace(0, 1)
# y_draw = - (ans[0] + ans[1] * x_draw) / ans[2]

x, y = generateData(100, 0.1, 0.7, 0.3)
m, n = np.shape(x)
x = np.c_[np.ones(m), x]
print(y.shape)
type1_x = []
type1_y = []
type2_x = []
type2_y = []
for i in range(len(x)):
    if y[i] == 1:
        type1_x.append(x[i][1])
        type1_y.append(x[i][2])
    else:
        type2_x.append(x[i][1])
        type2_y.append(x[i][2])
print(len(type1_x))
print(len(type2_x))
plt.scatter(type1_x, type1_y, facecolor="none",
            edgecolor="b", label="positive")
plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")
# plt.plot(x_draw, y_draw)
plt.legend()
plt.show()

ms = mushroom_read.MushroomProcessing()
mushroom_x, mushroom_y = ms.get_data()
mushroom_y = np.transpose([mushroom_y])
m, n = mushroom_x.shape
m_train = 6000
m_test = m - m_train
mushroom_x_train = np.c_[np.ones(m_train), mushroom_x[0:m_train, ]]
mushroom_x_test = np.c_[np.ones(m_test), mushroom_x[m_train:, ]]
mushroom_y_train = mushroom_y[0:m_train, ]
mushroom_y_test = mushroom_y[m_train:, ]
# print(mushroom_x.shape)
# print(mushroom_y.shape)
nwmu = newton_methon.NewtonMethod(
    mushroom_x_train, mushroom_y_train, np.zeros(n+1), 1e-4)
gdmu = gradient_descent_with_regulation.GradientDescentWithRegulation(
    mushroom_x_train, mushroom_y_train, np.zeros(n+1), 1e-5)
# ans = gdmu.gradient()
ans = nwmu.fitting()
print(nwmu.accuracy(mushroom_x_test, mushroom_y_test, ans))
# print(mushroom_x[0:6000,].shape)
# print(ans.shape)
# print(mushroom_y.shape)
