import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(data_dimension, number=100):
    if data_dimension is 2:
        mean = [-2, 2]
        cov = [[1, 0], [0, 0.01]]
    elif data_dimension is 3:
        mean = [1, 2, 3]
        cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        assert False
    sample_data = []
    for index in range(number):
        sample_data.append(np.random.multivariate_normal(mean, cov).tolist())
    return np.array(sample_data)


def draw_data(dimension_draw, origin_data, pca_data):
    if dimension_draw is 2:
        plt.scatter(origin_data[:, 0], origin_data[:, 1], facecolor="none", edgecolor="b", label="Origin Data")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], facecolor='r', label='PCA Data')
    elif dimension_draw is 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(origin_data[:, 0], origin_data[:, 1], origin_data[:, 2],
                   facecolor="none", edgecolor="b", label='Origin Data')
        ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], facecolor='r', label='PCA Data')
    else:
        assert False
    plt.legend()
    plt.show()


def pca(data, reduced_dimension):
    rows, columns = data.shape
    assert reduced_dimension <= columns
    x_mean = 1.0 / rows * np.sum(data, axis=0)
    decentralise_x = data - x_mean
    cov = decentralise_x.T.dot(decentralise_x)
    eigenvalues, feature_vectors = np.linalg.eig(cov)
    min_d = np.argsort(eigenvalues)
    for i in range(columns - reduced_dimension):
        feature_vectors = np.delete(feature_vectors, min_d[i], axis=1)
    return feature_vectors, x_mean


dimension = 2
data_number = 50
x = generate_data(dimension, number=data_number)
w, mu_x = pca(x, dimension - 1)
x_pca = (x - mu_x).dot(w).dot(w.T) + mu_x
print("Feature vectors:")
print(w)
print("Mean vector:")
print(mu_x)
draw_data(dimension, x, x_pca)
