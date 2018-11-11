import numpy as np
import os
import struct
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MNIST_DIMENSION = 784
MNIST_WIDTH = 28
MNIST_HEIGHT = 28


def generate_data(data_dimension, number=100):
    """ 生成2维或者3维数据 """
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


def load_mnist(path, kind='train'):
    """ 读取mnist中的训练数据 """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), MNIST_DIMENSION)
    return images, labels


def draw_data(dimension_draw, origin_data, pca_data):
    """ 将PCA前后的数据进行可视化对比 """
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
    """ 进行PCA(Principal Component Analysis)
    data:为原始数据
    reduced_dimension:为需要降低到的维数 """
    rows, columns = data.shape
    assert reduced_dimension <= columns
    x_mean = 1.0 / rows * np.sum(data, axis=0)
    decentralise_x = data - x_mean  # 去中心化
    cov = decentralise_x.T.dot(decentralise_x)  # 计算协方差
    eigenvalues, feature_vectors = np.linalg.eig(cov)  # 特征值分解
    min_d = np.argsort(eigenvalues)
    # 选取最大的特征值对应的特征向量
    feature_vectors = np.delete(feature_vectors, min_d[:columns - reduced_dimension], axis=1)
    return feature_vectors, x_mean


def psnr(source, target):
    """ 计算信噪比 """
    diff = source - target
    diff = diff ** 2
    rmse = np.sqrt(np.mean(diff))
    return 20 * np.log10(255.0 / rmse)


def psnr_trace(x_origin):
    """计算全部的信噪比"""
    ans = []
    for dim in range(1, MNIST_DIMENSION):
        w, mu = pca(x_origin, dim)
        x_pca = (x_origin - mu).dot(w).dot(w.T) + mu
        temp = np.abs(np.mean([psnr(x_origin[i], x_pca[i]) for i in range(len(x_origin))]))
        print(dim, temp)
        ans.append(temp)
    plt.plot(list(range(1, MNIST_DIMENSION)), ans)
    plt.show()


# 用于生成数据的测试
dimension = 3
data_number = 50
x = generate_data(dimension, number=data_number)
w, mu_x = pca(x, dimension - 1)
x_pca = (x - mu_x).dot(w).dot(w.T) + mu_x
print("Feature vectors:")
print(w)
print("Mean vector:")
print(mu_x)
draw_data(dimension, x, x_pca)

# 用于mnist数据集的测试
X_train, y_train = load_mnist('./mnist')
# psnr_trace(X_train)
d = 30
w_mnist, mu_mnist = pca(X_train, d)
x_pca_mnist = (X_train - mu_mnist).dot(w_mnist).dot(w_mnist.T) + mu_mnist
print("PSNR:")
print(np.abs(np.mean([psnr(X_train[i], x_pca_mnist[i]) for i in range(len(X_train))])))
x_pca_show = x_pca_mnist.astype(np.int)
fig, ax = plt.subplots(nrows=5, ncols=4, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(10):
    img = X_train[i].reshape(MNIST_HEIGHT, MNIST_WIDTH)
    ax[2 * i].imshow(img, cmap='Greys')
    img_compared = x_pca_show[i].reshape(MNIST_HEIGHT, MNIST_WIDTH)
    ax[2 * i + 1].imshow(img_compared, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.title('Dimension = ' + str(d))
plt.show()
