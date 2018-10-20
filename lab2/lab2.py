import numpy as np
import random
from matplotlib import pyplot as plt

import mushroom_read
import gradient_descent
import newton_method


def sigmod(z):
    return 1.0 / (1.0 + np.exp(z))


def generate_data(number, mean_pos, mean_neg, proportion_pos, scale1=0.1, scale2=0.2):
    """ 
    args:
        number:样本的数量
        mean_pos:正例的平均值
        mean_neg:反例的平均值
        proportion_pos:正例占样本的数量
        scale1:正例的高斯分布的方差
        scale2:反例的高斯分布的方差
     """
    assert (0 < proportion_pos < 1)
    x_sample = []
    y_sample = []
    number_pos = int(number * proportion_pos)
    number_neg = number - number_pos
    mean_pos_1 = mean_pos * (np.random.random_sample() + 0.5)
    mean_pos_2 = mean_pos * (np.random.random_sample() + 0.5)
    mean_neg_1 = mean_neg * np.random.random_sample()
    mean_neg_2 = mean_neg * np.random.random_sample()

    # mean_pos_1 = mean_pos * 0.9
    # mean_pos_2 = mean_pos * 1.1
    # mean_neg_1 = mean_neg * 1.1
    # mean_neg_2 = mean_neg * 0.9

    while True:
        if number_neg == 0 and number_pos == 0:
            break
        elif number_neg == 0:
            number_pos = number_pos - 1
            x_sample.append([random.gauss(mean_pos_1, scale1),
                             random.gauss(mean_pos_2, scale1)])
            y_sample.append(1)
        elif number_pos == 0:
            number_neg = number_neg - 1
            x_sample.append([random.gauss(mean_neg_1, scale2),
                             random.gauss(mean_neg_2, scale2)])
            y_sample.append(0)
        else:
            if np.random.randint(0, 2) == 0:
                number_neg = number_neg - 1
                x_sample.append([random.gauss(mean_neg_1, scale2),
                                 random.gauss(mean_neg_2, scale2)])
                y_sample.append(0)
            else:
                number_pos = number_pos - 1
                x_sample.append([random.gauss(mean_pos_1, scale1),
                                 random.gauss(mean_pos_2, scale1)])
                y_sample.append(1)
    return x_sample, np.array(y_sample)


def split_data(x_sample, y_sample, test_rate=0.3):
    number = len(x_sample)
    number_test = int(number * test_rate)
    number_train = number - number_test
    __x_train = []
    __x_test = []
    __y_train = []
    __y_test = []
    for i in range(number):
        if number_test > 0:
            if number_train == 0 or np.random.randint(2) == 0:
                number_test = number_test - 1
                __x_test.append(x_sample[i])
                __y_test.append(y_sample[i])
            else:
                number_train = number_train - 1
                __x_train.append(x_sample[i])
                __y_train.append(y_sample[i])
        else:
            number_train = number_train - 1
            __x_train.append(x_sample[i])
            __y_train.append(y_sample[i])
    return np.array(__x_train), np.array(__y_train), np.array(__x_test), np.array(__y_test)


def load_watermelon_data():
    """ 读取西瓜书3.3a的数据 """
    data_set = np.loadtxt("./watermelon-3.3.csv", delimiter=",")
    x_temp = data_set[:, 1:3]
    y_temp = data_set[:, 3]
    return x_temp, y_temp


def accuracy(__x_test, __y_test, beta):
    columns = len(__x_test)
    count = 0
    for index in range(columns):
        if sigmod(beta @ __x_test[index]) < 0.5 and __y_test[index] == 1:
            count = count + 1
        elif sigmod(beta @ __x_test[index]) > 0.5 and __y_test[index] == 0:
            count = count + 1
    return (1.0 * count) / columns


def draw_2_dimensions(x_sample, y_sample):
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(x_sample)):
        if y_sample[i] == 1:
            type1_x.append(x_sample[i][1])
            type1_y.append(x_sample[i][2])
        else:
            type2_x.append(x_sample[i][1])
            type2_y.append(x_sample[i][2])

    plt.scatter(type1_x, type1_y, facecolor="none", edgecolor="b", label="positive")
    plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")


# 用于西瓜书上样例的测试
watermelon_lambda = np.exp(-10)  # 西瓜书超参数

watermelon_x, watermelon_y = load_watermelon_data()
watermelon_rows, watermelon_columns = np.shape(watermelon_x)
watermelon_x = np.c_[np.ones(watermelon_rows), watermelon_x]
# 使用梯度下降法
gdr_watermelon = gradient_descent.GradientDescent(watermelon_x, watermelon_y,
                                                  np.zeros(watermelon_columns + 1), hyper=watermelon_lambda)
# 使用牛顿法
nw_watermelon = newton_method.NewtonMethod(watermelon_x, watermelon_y,
                                           np.zeros(watermelon_columns + 1), hyper=watermelon_lambda)

ans_nw_watermelon = nw_watermelon.fitting()
ans_gd_watermelon = gdr_watermelon.fitting()
print("GD watermelon:", ans_gd_watermelon)
print("NW watermelon:", ans_nw_watermelon)
x_draw_watermelon = np.linspace(0, 1)
y_draw_gd_watermelon = - (ans_gd_watermelon[0] + ans_gd_watermelon[1] * x_draw_watermelon) / ans_gd_watermelon[2]
y_draw_nw_watermelon = - (ans_nw_watermelon[0] + ans_nw_watermelon[1] * x_draw_watermelon) / ans_nw_watermelon[2]
plt.plot(x_draw_watermelon, y_draw_gd_watermelon, label="GD")
plt.plot(x_draw_watermelon, y_draw_nw_watermelon, label="NW")
draw_2_dimensions(watermelon_x, watermelon_y)
plt.legend()
plt.show()

# 用于生成数据的测试
gen_lambda = np.exp(-8)   # 惩罚项系数
number_gen = 100    # 样本数量
proportion_pos_gen = 0.5    # 正例比例
mean_gen_pos = 0.1  # 正例基础均值
mean_gen_neg = 0.8  # 反例基础均值
generating_x, generating_y = generate_data(number_gen, mean_gen_pos, mean_gen_neg, proportion_pos_gen)
generating_x = np.c_[np.ones(len(generating_x)), generating_x]
x_train_gen, y_train_gen, x_test_gen, y_test_gen = split_data(generating_x, generating_y)
generating_rows, generating_columns = np.shape(x_train_gen)
# 使用梯度下降进行测试
gdr_gen = gradient_descent.GradientDescent(x_train_gen, y_train_gen, np.zeros(generating_columns), hyper=gen_lambda)
ans_gdr_gen = gdr_gen.fitting()
# 使用牛顿法进行测试
nw_gen = newton_method.NewtonMethod(x_train_gen, y_train_gen, np.zeros(generating_columns), hyper=gen_lambda)
ans_nw_gen = nw_gen.fitting()
print("Generating GD:", ans_gdr_gen)  # 梯度下降法系数
print("Generating NW:", ans_nw_gen)   # 牛顿法系数

x_draw_gen = np.linspace(-0.4, 1)
y_draw_gdr_gen = - (ans_gdr_gen[0] + ans_gdr_gen[1] * x_draw_gen) / ans_gdr_gen[2]
y_draw_nw_gen = - (ans_nw_gen[0] + ans_nw_gen[1] * x_draw_gen) / ans_nw_gen[2]

plt.plot(x_draw_gen, y_draw_gdr_gen, label="GD")
plt.plot(x_draw_gen, y_draw_nw_gen, label="NW")

print("Generating GD accuracy:", accuracy(x_test_gen, y_test_gen, ans_gdr_gen))
print("Generating NW accuracy:", accuracy(x_test_gen, y_test_gen, ans_nw_gen))

draw_2_dimensions(generating_x, generating_y)
plt.legend()
plt.show()

# 用于UCI mushroom 测试
ms = mushroom_read.MushroomProcessing()
ms_lambda = np.exp(-8)  # mushroom 超参数
mushroom_x, mushroom_y = ms.get_data()
mushroom_x = np.c_[np.ones(len(mushroom_x)), mushroom_x]
x_train, y_train, x_test, y_test = split_data(mushroom_x, mushroom_y, test_rate=0.3)
mushroom_rows, mushroom_columns = x_train.shape
nw_mushroom = newton_method.NewtonMethod(x_train, y_train, np.zeros(mushroom_columns), hyper=ms_lambda)
mushroom_ans_nw = nw_mushroom.fitting()

gd_mushroom = gradient_descent.GradientDescent(x_train, y_train, np.zeros(mushroom_columns), hyper=ms_lambda)
mushroom_ans_gd = gd_mushroom.fitting()
print("Mushrooms GD accuracy:", accuracy(x_test, y_test, mushroom_ans_gd))
print("Mushrooms NW accuracy:", accuracy(x_test, y_test, mushroom_ans_nw))
