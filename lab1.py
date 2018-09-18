import numpy as np
import itertools
import functools
from matplotlib import pyplot as plt

# maybe 有的矩阵求逆函数

""" 
generate data
return X, T
"""
def generateData(number, scale=0.5):
    assert isinstance(number, int)
    assert number > 0
    assert scale > 0
    X = np.linspace(0, 1, num=number)
    T = np.sin(2 * np.pi * X) + np.random.normal(scale=scale, size=X.shape)
    return X, T


"""
假设X为一维数组
默认degree = 2
"""
def transform(X, degree=2):
    assert isinstance(degree, int)
    X_T = X.transpose()
    X = np.transpose([X])
    # print("X:\n",X)
    # print("X_T:\n", X_T)
    features = [np.ones(len(X))]
    for i in range(0, degree):
        # print(features[i])
        # print(type(np.ones(len(X))))
        # print(type(np.array(features[i])))
        features.append(np.multiply(X_T, features[i]))
        # print(features)
        # for items in itertools.combinations_with_replacement(X_T, degree):
        #     features.append(functools.reduce(lambda x, y: x * y, items))
        #     print(features)
        # print(features[degree])
    
    return np.asarray(features).transpose()
    # answer = [np.ones(len(X))]
    # print(X)
    # print(answer)
    # print(np.asarray(answer).transpose())


def fitting(X_training, T_training):
    W = np.dot(np.linalg.pinv(X_training), T_training)
    return W
def predict(X_Test, W):
    return np.dot(X_Test, W)


degree = 5
X_training, T_training = generateData(10)
W = fitting(transform(X_training, degree=degree), T_training)
X_test = np.linspace(0, 1, 100)
Y = np.sin(2 * np.pi * X_test)
T_test = predict(transform(X_test,degree=degree), W)

plt.scatter(X_training, T_training, facecolor="none", edgecolor="r")
plt.plot(X_test, Y, "g")
plt.plot(X_test, T_test, "b")

plt.show()

