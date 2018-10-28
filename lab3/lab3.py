import numpy as np
import random
from matplotlib import pyplot as plt
import collections

def watermelon_data():
    watermelon = np.array([[ 0.697  ,0.46 ],
                         [ 0.774  ,0.376],
                         [ 0.634  ,0.264],
                         [ 0.608  ,0.318],
                         [ 0.556  ,0.215],
                         [ 0.403  ,0.237],
                         [ 0.481  ,0.149],
                         [ 0.437  ,0.211],
                         [ 0.666  ,0.091],
                         [ 0.243  ,0.267],
                         [ 0.245  ,0.057],
                         [ 0.343  ,0.099],
                         [ 0.639  ,0.161],
                         [ 0.657  ,0.198],
                         [ 0.36   ,0.37 ],
                         [ 0.593  ,0.042],
                         [ 0.719  ,0.103],
                         [ 0.359  ,0.188],
                         [ 0.339  ,0.241],
                         [ 0.282  ,0.257],
                         [ 0.748  ,0.232],
                         [ 0.714  ,0.346],
                         [ 0.483  ,0.312],
                         [ 0.478  ,0.437],
                         [ 0.525  ,0.369],
                         [ 0.751  ,0.489],
                         [ 0.532  ,0.472],
                         [ 0.473  ,0.376],
                         [ 0.725  ,0.445],
                         [ 0.446  ,0.459]])
    return watermelon

def generate_data(means, number, k):
    """ 生成2维数据 """
    assert len(means) == k
    assert len(number) == k
    cov = [[0.1, 0], [0, 0.1]]
    data = []
    for i in range(k):
        for j in range(number[i]):
            temp_x1, temp_x2 = np.random.multivariate_normal(
                [means[i], means[i]], cov
            )
            data.append([temp_x1, temp_x2])
    return np.array(data)

def euclidean_distance(x1, x2):
    """ 输出2个向量之间的欧式距离 """
    assert x1.shape == x2.shape
    return np.linalg.norm(x1 - x2)

def k_means(training_x, k, delta=1e-6):
    data_rows = training_x.shape[0]
    mu = training_x[random.sample(range(data_rows), k)]
    times = 0
    while True:
        c = collections.defaultdict(list)
        for i in range(data_rows):
            dij = [euclidean_distance(training_x[i], mu[j]) for j in range(k)]
            lambda_j = np.argmin(dij)
            c[lambda_j].append(training_x[i].tolist())

        # for i in range(k):
        #     mu[i] = np.mean(c[i], axis=0)
        new_mu = np.array([np.mean(c[i],axis=0).tolist() for i in range(k)])

        loss = 0
        for i in range(k):
            loss += euclidean_distance(np.array(mu[i]), np.array(new_mu[i]))
        if loss > delta:
            mu = new_mu
        else:
            break

        # print(times)
        # times = times + 1
        # for i in range(k):
        #     print(mu[i])
    return mu, c

k = 3
means = [4, -4, 0]
number = [100, 100, 100]
data = generate_data(means, number, k)
mu, c = k_means(data, k)
# ans = []
# TODO k-means 可能会陷入局部最优解 所以如何选择初始簇中心点
# 参考 https://blog.csdn.net/zhihaoma/article/details/48649489?utm_source=blogxgwz0
# 可以选择彼此距离尽可能远的k个点 作为初始簇中心
# for times in range(200):
#     print(times)
#     mu, c = k_means(data, k)
#     summ = 0
#     for i in range(k):
#         temp = [euclidean_distance(np.array(c[i][j]), mu[i]) for j in range(len(c[i]))]
#         summ += np.sum(temp)
#     print(summ)
#     ans.append(summ)
# print(ans)
# print(np.min(ans))
# print(np.max(ans))

# watermelon = watermelon_data()
# mu, c = k_means(watermelon, k)
plt.scatter(mu[:,0], mu[:,1], facecolor="none", edgecolor="b", label="center")
for i in range(k):
    plt.scatter(np.array(c[i])[:,0], np.array(c[i])[:,1], marker="x", label=str(i+1))
plt.legend()
plt.show() 
