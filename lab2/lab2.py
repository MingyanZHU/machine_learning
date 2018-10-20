import mushroom_read
import numpy as np
import random
from matplotlib import pyplot as plt

import gradient_descent
import gradient_descent_with_regulation
import newton_methon


def sigmod(z):
    return 1.0 / (1.0 + np.exp(z))


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
    number_neg = number - number_pos
    meanPos_1 = meanPos * (np.random.random_sample() + 0.5)
    meanPos_2 = meanPos * (np.random.random_sample() + 0.5)
    meanNeg_1 = meanNeg * np.random.random_sample()
    meanNeg_2 = meanNeg * np.random.random_sample()

    # meanPos_1 = meanPos * 0.9
    # meanPos_2 = meanPos * 1.1
    # meanNeg_1 = meanNeg * 1.1
    # meanNeg_2 = meanNeg * 0.9

    while True:
        if number_neg == 0 and number_pos == 0:
            break
        elif number_neg == 0:
            number_pos = number_pos - 1
            x.append([random.gauss(meanPos_1, scale1),
                      random.gauss(meanPos_2, scale1)])
            y.append(1)
        elif number_pos == 0:
            number_neg = number_neg - 1
            x.append([random.gauss(meanNeg_1, scale2),
                      random.gauss(meanNeg_2, scale2)])
            y.append(0)
        else:
            if np.random.randint(0, 2) == 0:
                number_neg = number_neg - 1
                x.append([random.gauss(meanNeg_1, scale2),
                          random.gauss(meanNeg_2, scale2)])
                y.append(0)
            else:
                number_pos = number_pos - 1
                x.append([random.gauss(meanPos_1, scale1),
                          random.gauss(meanPos_2, scale1)])
                y.append(1)
    return x, np.array(y)


def split_data(x, y, test_rate=0.3):
    number = len(x)
    number_test = int(number * test_rate)
    number_train = number - number_test
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(number):
        if number_test > 0:
            if number_train == 0 or np.random.randint(2) == 0:
                number_test = number_test - 1
                x_test.append(x[i])
                y_test.append(y[i])
            else:
                number_train = number_train - 1
                x_train.append(x[i])
                y_train.append(y[i])
        else:
            number_train = number_train - 1
            x_train.append(x[i])
            y_train.append(y[i])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def loadWaterMelonData():
    """ 读取西瓜书3.3a的数据 """
    dataSet = np.loadtxt("./watermelon-3.3.csv", delimiter=",")
    x = dataSet[:, 1:3]
    y = dataSet[:, 3]
    return x, y


def accuracy(x_test, y_test, beta):
    m = len(x_test)
    count = 0
    for i in range(m):
        if sigmod(beta @ x_test[i]) < 0.5 and y_test[i] == 1:
            count = count + 1
        elif sigmod(beta @ x_test[i]) > 0.5 and y_test[i] == 0:
            count = count + 1
    return (1.0 * count) / m

# # 用于西瓜书上样例的测试
# x, y = loadWaterMelonData()
# m, n = np.shape(x)
# x = np.c_[np.ones(m), x]
# print(y.shape)
# gdr = gradient_descent_with_regulation.GradientDescentWithRegulation(
#     x, y, np.zeros(n+1), np.exp(-7),rate=0.01)
# nw = newton_methon.NewtonMethod(x, y, np.zeros(n + 1), np.exp(-7))

# ans_nw = nw.fitting()
# ans_gd = gdr.gradient()
# print("GD:", ans_gd)
# print("NW:", ans_nw)
# x_draw = np.linspace(0, 1)
# y_draw_gd = - (ans_gd[0] + ans_gd[1] * x_draw) / ans_gd[2]
# y_draw_nw = - (ans_nw[0] + ans_nw[1] * x_draw) / ans_nw[2]
# plt.plot(x_draw, y_draw_gd, label="GD")
# plt.plot(x_draw,y_draw_nw, label="NW")
# type1_x = []
# type1_y = []
# type2_x = []
# type2_y = []
# for i in range(len(x)):
#     if y[i] == 1:
#         type1_x.append(x[i][1])
#         type1_y.append(x[i][2])
#     else:
#         type2_x.append(x[i][1])
#         type2_y.append(x[i][2])
# 
# plt.scatter(type1_x, type1_y, facecolor="none",
#             edgecolor="b", label="positive")
# plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")
# plt.legend()
# plt.show()


x, y = generateData(100, 0.1, 0.8, 0.5)
x = np.c_[np.ones(len(x)), x]
x_train, y_train, x_test, y_test = split_data(x, y)
# # print(x_test)
# # print(y_test)
m, n = np.shape(x_train)

gdr = gradient_descent_with_regulation.GradientDescentWithRegulation(x_train, y_train, np.zeros(n), np.exp(-9), rate=0.01)
ans = gdr.gradient()

print(ans)
print(gdr.predict(ans))

x_draw = np.linspace(-0.4, 1)
y_draw = - (ans[0] + ans[1] * x_draw) / ans[2]
plt.plot(x_draw, y_draw)

print(accuracy(x_test, y_test, ans))

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

plt.scatter(type1_x, type1_y, facecolor="none",
            edgecolor="b", label="positive")
plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")
plt.legend()
plt.show()

# ms = mushroom_read.MushroomProcessing()
# mushroom_x, mushroom_y = ms.get_data()
# mushroom_x = np.c_[np.ones(len(mushroom_x)), mushroom_x]
# x_train, y_train, x_test, y_test = split_data(mushroom_x, mushroom_y, test_rate=0.4)
# m, n = x_train.shape
# nw_mushroom = newton_methon.NewtonMethod(x_train, y_train, np.zeros(n), np.exp(-8))
# ans = nw_mushroom.fitting()

# # gd_mushroom = gradient_descent_with_regulation.GradientDescentWithRegulation(x_train, y_train, np.zeros(n), np.exp(-8))
# # ans = gd_mushroom.gradient()
# print(accuracy(x_test, y_test, ans))
