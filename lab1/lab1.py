import numpy as np
from matplotlib import pyplot as plt
import newton_method
import analytical_solution
import gradient_descent
import conjugate_gradient


def generateData(number, scale=0.5):
    """ Generate training or test data.
    Args:
        number: data number you want which is an integer
        scale: the variance of Gaussian diribution for noise.
    Returns:
        X: a one-dimensional array containing all uniformly distributed x.
        T: sin(2 * pi * x) with Gaussian distribution noise with variance of scale. 
    """
    assert isinstance(number, int)
    assert number > 0
    assert scale > 0
    X = np.linspace(0, 1, num=number)
    T = np.sin(2 * np.pi * X) + np.random.normal(scale=scale, size=X.shape)
    return X, T


def transform(X, degree=2):
    """
    Transform an array to (len(X), degree + 1) matrix.
    Args:
        X: an ndarray.
        degree:int, degree for polynomial.
    Returns:
        for example, [a b] -> [[1 a a^2] [1 b b^2]]
    """
    assert isinstance(degree, int)
    assert X.ndim == 1
    X_T = X.transpose()
    X = np.transpose([X])
    features = [np.ones(len(X))]
    for i in range(0, degree):
        features.append(np.multiply(X_T, features[i]))

    return np.asarray(features).transpose()


def predict(X_Test, w):
    return np.dot(X_Test, w)


number_train = 10  # 训练样本的数量
number_test = 100  # 测试样本的数量
degree = 9  # 多项式的阶数
X_training, T_training = generateData(number_train, scale=0.3)
X_test = np.linspace(0, 1, number_test)
X_Train = transform(X_training, degree=degree)
X_Test = transform(X_test, degree=degree)
Y = np.sin(2 * np.pi * X_test)


title = "degree = " + str(degree) + ", number_train = " + \
    str(number_train) + ", number_test = " + str(number_test)
plt.title(title)
plt.ylim(-1.5, 1.5)

# # 用于解析解(不带正则项)的实验
# plt.scatter(X_training, T_training, facecolor="none",
#             edgecolor="b", label="training data")
# anaSolution = analytical_solution.AnalyticalSolution(X_Train, T_training)
# plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
# plt.plot(X_test, predict(X_Test, anaSolution.fitting()),
#          "r", label="analytical solution")
# plt.legend()
# plt.show()


# # 用于解析解(带正则项)的实验 寻找最优的超参数
# # 经过100次实验 最终得到的最优参数为e^-7
# anaSolution = analytical_solution.AnalyticalSolution(X_Train, T_training)
# hyperTestList = []
# hyperTrainList = []
# hyperList = range(-50, 1)
# for hyper in hyperList:
#     w_analytical_with_regulation = anaSolution.fitting_with_regulation(
#         np.exp(hyper))
#     T_test = predict(X_Test, w_analytical_with_regulation)
#     hyperTestList.append(anaSolution.E_rms(T_test, Y))
#     hyperTrainList.append(anaSolution.E_rms(T_training, predict(
#         transform(X_training, degree=degree), w_analytical_with_regulation)))
# bestHyper = hyperList[np.where(hyperTestList ==
#                                np.min(hyperTestList))[0][0]]
# print("bestHyper:", bestHyper, np.min(hyperTestList))
# annotate = "$\lambda = e^{" + str(bestHyper) + "}$"
# plt.title(title)
# plt.ylabel("$E_{RMS}$")
# plt.ylim(0, 1)
# plt.xlabel("$ln \lambda$")
# plt.annotate(annotate, xy=(-30, 0.3))
# plt.plot(hyperList, hyperTestList, 'o-', mfc="none", mec="b", ms=5,
#          label="Test")
# plt.plot(hyperList, hyperTrainList, 'o-',
#          mfc="none", mec="r", ms=5, label="Training")
# plt.legend()
# plt.show()


# # 此处用于确认带有惩罚项的解析解的正确性实验
# bestHyper = -7 #此处的最佳的超参数是经过上面提到的实验中确定的
# anaSolution = analytical_solution.AnalyticalSolution(X_Train, T_training)
# w_analytical_with_regulation = anaSolution.fitting_with_regulation(np.exp(bestHyper))
# T_test = predict(X_Test, w_analytical_with_regulation)
# annotate = "$\lambda = e^{" + str(bestHyper) + "}$"
# plt.ylim(-1.5, 1.5)
# plt.scatter(X_training, T_training, facecolor="none",
#             edgecolor="b", label="training data")
# plt.plot(X_test, T_test, "r", label="analytical with regulation")
# plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
# plt.annotate(annotate, xy=(0.3, -0.5))
# plt.legend()
# plt.show()

""" TODO 牛顿法与共轭梯度与梯度下降的对比
    TODO 重构下面的代码
"""

# # 用于测试共轭梯度与梯度下降法
bestHyper = -7  # 此处的最佳的超参数是经过上面提到的实验中确定的
# w_analytical_with_regulation = fitting_with_regulation(
#     X_Train, T_training, hyper=np.exp(bestHyper))
# T_test = predict(X_Test, w_analytical_with_regulation)
w_0 = np.zeros(degree+1)
# w_gradient = gradient_rate(X_Train, T_training, bestHyper, w_0)
# w_conjugate = conjugate_gradient(X_Train, T_training, bestHyper, w_0)
# print()
# print("w_analytical_with_regulation(Analytical solution):\n",
#       w_analytical_with_regulation)
# print("w_gradient(Gradient descent):\n", w_gradient)
# print("w_conjugate(Conjugate gradient):\n", w_conjugate)

# title = "degree = " + str(degree) + ", number_train = " + \
#     str(number_train) + ", number_test = " + str(number_test)
# plt.figure(figsize=(15, 6))
# plt.subplot(121)
# plt.title(title)
# plt.ylim(-1.5, 1.5)
plt.scatter(X_training, T_training, facecolor="none",
            edgecolor="b", label="training data")
# plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
# plt.plot(X_test, T_test, "r", label="Analytical with regulation")
# plt.plot(X_test, predict(X_Test, w_gradient), "c", label="Gradient descent")
# plt.legend()

# plt.subplot(122)
# plt.ylim(-1.5, 1.5)
# plt.title(title)
# plt.scatter(X_training, T_training, facecolor="none",
#             edgecolor="b", label="training data")
# plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
# plt.plot(X_test, T_test, "r", label="Analytical regulation")
# plt.plot(X_test, predict(X_Test, w_conjugate), "m",
#          label="Conjugate gradient")
# plt.legend()
# plt.show()

