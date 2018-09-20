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

def fitting_with_regulation(X_training, T_training, hyperparameter = np.exp(-18)):
    X_T = X_training.transpose()
    # W = np.linalg.pinv(np.dot(X_T, X_training) + np.eye(len(X_T)) * hyperparameter) @ X_T @ T_training
    W = np.linalg.solve(np.eye(len(X_T)) * hyperparameter + np.dot(X_T, X_training), np.dot(X_T, T_training))
    return W

def predict(X_Test, W):
    return np.dot(X_Test, W)


degree = 9
X_training, T_training = generateData(20)
X_test = np.linspace(0, 1, 100)
X_Train = transform(X_training, degree=degree)
X_Test = transform(X_test, degree=degree)
Y = np.sin(2 * np.pi * X_test)

list = []
hyperList = range(-30, 1)

for hyper in hyperList:
    print("hyper = ", hyper)
    W = fitting_with_regulation(X_Train, T_training, hyperparameter=np.exp(hyper))
    T_test = predict(transform(X_test,degree=degree), W)
    # print(W)
    error = Y - T_test
    ans = np.mean(error @ np.transpose([error]))
    # print(hyper, ans)
    list.append(ans)
    
# plt.figure(figsize=(15, 6))

bestHyper = hyperList[np.where(list==np.min(list))[0][0]]
print(bestHyper, np.min(list))
# plt.subplot(121)
# plt.plot(hyperList, list)
# plt.show()

W = fitting_with_regulation(X_Train, T_training, hyperparameter=np.exp(bestHyper))
T_test = predict(X_Test, W)
# plt.subplot(122)
# plt.scatter(X_training, T_training, facecolor="none", edgecolor="r")
# plt.ylim(-1.5, 1.5)
# plt.plot(X_test, Y, "g")
# plt.plot(X_test, T_test, "b")

# plt.show()
W_t = np.zeros(degree+1)
# print(X_Train.transpose() @ X_Train @ W_t)
# print(X_Train.transpose() @ X_Train @ np.transpose([W_t]))
# print(W_t)
# print(X_training)
def h(X_Train, T_training, hyper, W_t):
    X_T = X_Train.transpose()
    return X_T @ X_Train @ W_t - X_T @ T_training + W_t * np.exp(hyper)
def E(X_Train, T_training, hyper, W_t):
    W_T = np.transpose([W_t])
    X_T = X_Train.transpose()
    return 0.5 * np.linalg.norm(X_Train @ W_T - T_training) ** 2 + 0.5 * np.exp(hyper) * W_t @ W_T
    # return np.dot(np.dot(X_T, X_Train) + np.eye(len(X_T)) * hyper, W_t) - np.dot(X_T, T_training)
# print(h(X_Train, T_training, bestHyper, W_t))
def gradient_descent(X_Train, T_training, hyper, W_t, rate=0.01):
    error = E(X_Train, T_training, hyper, W_t)
    for i in range(1000000):
        temp = h(X_Train, T_training, hyper, W_t)
        # print(temp)
        w = W_t - rate * temp
        # print(w)
        # error0 = np.linalg.norm(E(X_Train, T_training, hyper, W_t)-E(X_Train, T_training, hyper, w))
        error0 = E(X_Train, T_training, hyper, w)
        # print(error0.shape)
        if np.abs(error0[0] - error[0]) < 0.000001:
            break
        # elif error0 > error:
        #     print("error")
        #     break
        else:
            print(i)
            print("abs:", np.abs(error - error0))
            print("error:", error)
            error = error0
            W_t = w
    return w
w = gradient_descent(X_Train, T_training, bestHyper, W_t)
print("W:", W)
print("w:", w)
# w = np.ones(4)
# print(h(X_Train, T_training, bestHyper, w))

# plt.subplot(121)
plt.scatter(X_training, T_training, facecolor="none", edgecolor="r")
plt.ylim(-1.5, 1.5)
plt.plot(X_test, Y, "g")
plt.plot(X_test, T_test, "b")

# plt.subplot(122)
plt.plot(X_test, predict(X_Test, w), "r")
plt.show()