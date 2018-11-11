import numpy as np
from matplotlib import pyplot as plt

import k_means
import gaussian_mixture_model
import iris_read


def watermelon_data():
    watermelon = np.array([[0.697, 0.46],
                           [0.774, 0.376],
                           [0.634, 0.264],
                           [0.608, 0.318],
                           [0.556, 0.215],
                           [0.403, 0.237],
                           [0.481, 0.149],
                           [0.437, 0.211],
                           [0.666, 0.091],
                           [0.243, 0.267],
                           [0.245, 0.057],
                           [0.343, 0.099],
                           [0.639, 0.161],
                           [0.657, 0.198],
                           [0.36, 0.37],
                           [0.593, 0.042],
                           [0.719, 0.103],
                           [0.359, 0.188],
                           [0.339, 0.241],
                           [0.282, 0.257],
                           [0.748, 0.232],
                           [0.714, 0.346],
                           [0.483, 0.312],
                           [0.478, 0.437],
                           [0.525, 0.369],
                           [0.751, 0.489],
                           [0.532, 0.472],
                           [0.473, 0.376],
                           [0.725, 0.445],
                           [0.446, 0.459]])
    return watermelon


def generate_data(sample_means, sample_number, number_k):
    """ 生成2维数据
    :argument number_k k类
    :argument sample_means k类数据的均值 以list的形式给出 如[[1, 2],[-1, -2], [0, 0]]
    :argument sample_number k类数据的数量 以list的形式给出 如[10, 20, 30]
    """
    assert number_k > 0
    assert len(sample_means) == number_k
    assert len(sample_number) == number_k
    cov = [[0.1, 0], [0, 0.1]]
    sample_data = []
    for index in range(number_k):
        for times in range(sample_number[index]):
            sample_data.append(np.random.multivariate_normal(
                [sample_means[index][0], sample_means[index][1]], cov).tolist())
    return np.array(sample_data)


k = 3
means = [[2, 4], [0, -4], [-2, 2]]
number = [100, 100, 100]
data = generate_data(means, number, k)

km = k_means.KMeans(data, k)
mu_random, c_random = km.k_means_random_center()
mu_normal, c_normal = km.k_means_not_random_center()
# watermelon = watermelon_data()
# mu, c = k_means(watermelon, k)
plt.subplot(131)
plt.title("K-Means:randomly")
for i in range(k):
    plt.scatter(np.array(c_random[i])[:, 0], np.array(c_random[i])[:, 1], marker="x", label=str(i + 1))
plt.scatter(mu_random[:, 0], mu_random[:, 1], facecolor="none", edgecolor="r", label="center")
plt.legend()

plt.subplot(132)
plt.title("K-Means:max distances")
for i in range(k):
    plt.scatter(np.array(c_normal[i])[:, 0], np.array(c_normal[i])[:, 1], marker="x", label=str(i + 1))
plt.scatter(mu_normal[:, 0], mu_normal[:, 1], facecolor="none", edgecolor="r", label="center")
plt.legend()

gmm = gaussian_mixture_model.GaussianMixtureModel(data, k=k)
mu_gmm, c_gmm = gmm.predict()

plt.subplot(133)
plt.title("GMM")
for i in range(k):
    plt.scatter(np.array(c_gmm[i])[:, 0], np.array(c_gmm[i])[:, 1], marker="x", label=str(i + 1))
plt.scatter(mu_gmm[:, 0], mu_gmm[:, 1], facecolor="none", edgecolor="r", label="center")
plt.legend()

plt.show()

iris = iris_read.IrisProcessing()
iris_data = iris.get_data()
gmm_iris = gaussian_mixture_model.GaussianMixtureModel(iris_data, k)
mu_iris, c_iris = gmm_iris.predict()
print(mu_iris)

km_iris = k_means.KMeans(iris_data, 3)
km_mu_iris, km_c_iris = km_iris.k_means_not_random_center()
print(km_mu_iris)
print(iris.acc(gmm_iris.sample_assignments))
print(iris.acc(km_iris.sample_assignments))
# TODO 对iris数据集，使用GMM得到的正确率低于使用k-means正确率
