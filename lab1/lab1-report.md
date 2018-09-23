<br/>
<br/>
<center> <font size = 5> 哈尔滨工业大学计算机科学与技术学院 </font></center>
<br/>
<br/>
<center> <font size = 7> 实验报告 </font></center>
<br/>
<br/>
<br/>
<center> <font size = 5> 
课程名称：机器学习 <br/>
课程类型：必修  <br/>
实验题目：多项式拟合正弦函数 
</font></center>
<br/>
<br/>
<center> <font size = 4> 学号：1160300314 </font></center>
<center> <font size = 4> 姓名：朱明彦 </font></center>

<div STYLE="page-break-after: always;"></div>
<!-- 此处用于换行 -->

# 一、实验目的
掌握最小二乘法求解（无惩罚项的损失函数），掌握增加惩罚项（2范数）的损失函数优化，梯度下降法、共轭梯度法，理解过拟合、客服过拟合的方法（如增加惩罚项、增加样本）。
# 二、实验要求及实验环境

## 实验要求
1. 生成数据，加入噪声；
2. 用高阶多项式函数拟合曲线；
3. 用解析解求解两种loss的最优解（无正则项和有正则项）
4. 优化方法求解最优解（梯度下降，共轭梯度）；
5. 用你得到的实验数据，解释过拟合。
6. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
7. 语言不限，可以用matlab，python。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如pytorch，tensorflow的自动微分工具。

## 实验环境
- OS: Microsoft Windows 10.0.17134
- Python 3.6.4
# 三、设计思想(本程序中用到的主要算法及数据结构)

## 算法原理

### 1、生成数据算法
主要是利用$sin(2\pi x)$函数产生样本，其中$x$均匀分布在$[0, 1]$之间，对于每一个目标值$t=sin(2\pi x)$增加一个$0$均值，方差为$0.5$的高斯噪声。

### 2、利用高阶多项式函数拟合曲线(不带惩罚项)
利用训练集合，对于每个新的$\hat{x}$，预测目标值$\hat{t}$。采用多项式函数进行学习，即利用式$(1)$来确定参数$w$，假设阶数$m$已知。
$$ y(x, w) = w_0 + w_1x + \dots + w_mx^m = \sum_{i = 0}^{m}w_ix^i \tag{1}$$
采用最小二乘法，即建立误差函数来测量每个样本点目标值$t$与预测函数$y(x, w)$之间的误差，误差函数即式$(2)$
$$ E(\bold{w}) = \frac{1}{2} \sum_{i = 1}^{N} \{y(x_i, \bold{w}) - t_i\}^2 \tag{2}$$
将上式写成矩阵形式如式$(3)$
$$ E(\bold{w}) = \frac{1}{2} (\bold{Xw} - \bold{T})'(\bold{Xw} - \bold{T})\tag{3}$$
其中
$$\bold{X} =
\left[
\begin{matrix}
 1      & x_1      & \cdots & x_1^m      \\
 1      & x_2      & \cdots & x_2^m      \\
 \vdots & \vdots & \ddots & \vdots \\
 1      & x_N      & \cdots & x_N^m      \\
\end{matrix}
\right], 
\bold{w} = 
\left[
\begin{matrix}
    w_0 \\ w_1 \\ \vdots \\ w_m
\end{matrix}
\right],
\bold{T} = 
\left[
\begin{matrix}
   t_1 \\ t_2 \\ \vdots \\ t_N 
\end{matrix}
\right]
$$

通过将上式求导我们可以得到式$(4)$
<!-- 很奇怪此处只有$$\cfrac{}{}$$这种格式可以正常在Chrome中打开，其他形式\frac或者\dfrac都没法显示中间的横线 -->
$$ \cfrac{\partial E}{\partial \bold{w}} = \bold{X'Xw} - \bold{X'T} \tag{4}$$
令 $\cfrac{\partial E}{\partial \bold{w}}=0$ 我们有式$(5)$即为$\bold{w^*}$
$$ \bold{w^*} = (\bold{X'X})^{-1}\bold{X'T} \tag{5}$$

### 3、带惩罚项的多项式函数拟合曲线
在不带惩罚项的多项式拟合曲线时，在参数多时$\bold{w}^*$具有较大的绝对值，本质就是发生了过拟合。对于这种过拟合，我们可以通过在优化目标函数式$(3)$中增加$\bold{w}$的惩罚项，因此我们得到了式$(6)$。

$$ \widetilde{E}(\bold{w}) = \frac{1}{2} \sum_{i=1}^{N} \{y(x_i, \bold{w}) - t_i\}^2 + \cfrac{\lambda}{2}|| \bold{w} || ^ 2 \tag{6}$$

同样我们可以将式$(6)$写成矩阵形式， 我们得到式$(7)$

$$ \widetilde{E}(\bold{w}) = \frac{1}{2}[(\bold{Xw} - \bold{T})'(\bold{Xw} - \bold{T}) + \lambda \bold{w'w}]\tag{7}$$

对式$(7)$求导我们得到式$(8)$

$$ \cfrac{\partial \widetilde{E}}{\partial \bold{w}} = \bold{X'Xw} - \bold{X'T} + \lambda\bold{w} \tag{8}$$

令 $\cfrac{\partial \widetilde{E}}{\partial \bold{w}} = 0$ 我们得到$\bold{w^*}$即式$(9)$，其中$\bold{I}$为单位阵。

$$ \bold{w^*} = (\bold{X'X} + \lambda\bold{I})^{-1}\bold{X'T}\tag{9}$$

### 4、梯度下降法求解最优解
对于$f(\bold{x})$如果在$\bold{x_i}$点可微且有定义，我们知道顺着梯度 $\nabla f(\bold{x_i})$为增长最快的方向，因此梯度的反方向 $-\nabla f(\bold{x_i})$ 即为下降最快的方向。因而如果有式$(10)$对于 $\alpha > 0$ 成立,
$$\bold{x_{i+1}}= \bold{x_i} - \alpha \nabla f(\bold{x_i}) \tag{10}$$
那么对于序列$\bold{x_0}, \bold{x_1}, \dots$ 我们有
$$ f(\bold{x_0}) \ge f(\bold{x_1}) \ge \dots$$
因此，如果顺利我们可以得到一个 $f(\bold{x_n})$ 收敛到期望的最小值，**当然对于我们此次实验很大可能性可以收敛到最小值**。

### 5、共轭梯度法求解最优解
共轭梯度法解决的主要是形如$\bold{Ax} = \bold{b}$的线性方程组解的问题，其中$\bold{A}$必须是对称的、正定的。
求解的方法就是我们先猜一个解$\bold{x_0}$，然后取梯度的反方向 $\bold{p_0} = \bold{b} - \bold{Ax}$，在n维空间的基中$\bold{p_0}$要与其与的基共轭并且为初始残差。

然后对于第k步的残差 $\bold{r_k} = \bold{b} - \bold{Ax}$，$\bold{r_k}$为$\bold{x} = \bold{x_k}$时的梯度反方向。由于我们仍然需要保证 $\bold{p_k}$ 彼此共轭。因此我们通过当前的残差和之前所有的搜索方向来构建$\bold{p_k}$，得到式$(11)$

$$ \bold{p_k} = \bold{r_k} - \sum_{i < k}\cfrac{\bold{p_i}^T\bold{Ar_k}}{\bold{p_i}^T\bold{Ap_i}}\bold{p_i}\tag{11}$$

进而通过当前的搜索方向$\bold{p_k}$得到下一步优化解$\bold{x_{k+1}} = \bold{x_k} + \alpha_k\bold{p_k}$，其中

$$ \alpha_k = \cfrac{\bold{p_k}^T(\bold{b} - \bold{Ax_k})}{\bold{p_k}^T\bold{Ap_k}} = \cfrac{\bold{p_k}^T\bold{r_k}}{{\bold{p_k}^T\bold{Ap_k}}}$$

## 算法的实现
对于数据生成、求解析解（有无正则项）都是可以利用numpy中的矩阵求逆等库变相降低了算法实现的难度，因此在这里就不再赘述，下面主要讲梯度下降法和共轭梯度法的算法实现。

### 1、梯度下降
此处我们利用带惩罚项的优化函数进行梯度下降法的实现。
由梯度下降主要解决的是如式$(12)$的线性方程组的解。

$$ \bold{Ax} - \bold{b} = 0 \tag{12}$$

另由式$(8)$我们可以得到式$(13)$

$$ J(\bold{w}) = (\bold{X'X} + \lambda\bold{I})\bold{w} - \bold{X'T}\tag{13}$$

联合式$(13)$与式$(12)$我们可以有：
$$ 
\left\{
         \begin{array}{lr}
         \bold{A} = \bold{X'X} + \lambda\bold{I} &  \\
         \bold{b} = \bold{X'T}  
         \end{array}
\right.\tag{14}
$$
进而我们实现算法如下，其中$\delta$为精度要求，通常可以设置为$\delta = 1\times10^{-6}$：
$$ 
\begin{array}{ll}
rate = 0.01, k = 0 \\
\bold{w_0} = \bold{0} \\
\widetilde{E}(\bold{w}) = \frac{1}{2N}[(\bold{Xw} - \bold{T})'(\bold{Xw} - \bold{T}) + \lambda \bold{w'w}]\\
J(\bold{w}) = (\bold{X'X} + \lambda\bold{I})\bold{w} - \bold{X'T} \\
\bold{loss_0} = \widetilde{\bold{E}}(\bold{w_0})\\
\bold{repeat:} \\
\quad \quad \bold{w_{k+1}} = \bold{w_k} - rate * J(\bold{w_k})\\
\quad \quad \bold{loss_{k+1}} = \widetilde{\bold{E}}(\bold{w_{k+1}})\\
\quad \quad \bold{if} \: \: \bold{abs}(\bold{loss_{k+1}} - \bold{loss_k}) < \delta \bold{\ then \ break \ loop}\\
\quad \quad k = k + 1\\
\bold{end \ repeat}\\
\end{array}
$$
### 2、共轭梯度下降
对于共轭梯度下降，算法实现如下，参考[wiki](https://en.wikipedia.org/wiki/Conjugate_gradient_method)实现：
<center>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/28f4e9c1591f48a96a5e9c1084b2be818fd8ea2a"/>
</center>

其中的$\bold{b}, \bold{A}$均与式$(14)$相同。


# 四、实验结果分析

**对于下面的所有的实验，在没有特别强调的情况下，所有训练样本的噪声标准差均为0.5。**

## 1、不带惩罚项的解析解

### (1)固定训练样本的大小为10，分别使用不同多项式阶数，测试的结果如下图。
<center>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_1_number_training_10_number_test_100.png" width="45%" >
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_3_number_training_10_number_test_100.png" width="45%">
</figure>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_5_number_training_10_number_test_100.png" width="45%">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_9_number_training_10_number_test_100.png" width="45%">
</figure>
</center>

我们可以看到在固定训练样本的大小之后，在多项式阶数为3时的拟合效果已经很好。继续提高多项式的阶数，**尤其在阶数为9的时候曲线“完美的”经过了所有的节点，这种剧烈的震荡并没有很好的拟合真实的背后的函数$sin(2\pi x)$，反而将所有噪声均很好的拟合，即表现出来一种过拟合的情况。** 
其出现过拟合的本质原因是，在阶数过大的情况下，模型的复杂度和拟合的能力都增强，因此可以通过过大或者过小的系数来实现震荡以拟合所有的数据点，以至于甚至拟合了所有的噪声。在这里由于我们的数据样本大小只有10，所以在阶数为9的时候，其对应的系数向量$\bold{w}$恰好有唯一解，因此可以穿过所有的样本点。

对于过拟合我们可以通过增加样本的数据或者通过增加惩罚项的方式来解决。增加数据集样本数量，使其超过参数向量的大小，就会在一定程度上解决过拟合问题。

### (2)固定多项式阶数，使用不同数量的样本数据，测试的结果如下图。
<center>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_9_number_training_10_number_test_100.png" width="45%" >
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_9_number_training_20_number_test_100.png" width="45%">
</figure>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_9_number_training_50_number_test_100.png" width="45%">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_9_number_training_100_number_test_100.png" width="45%">
</figure>
</center>

我们可以看到在固定多项式阶数为9的情况下，随着样本数量逐渐增加，过拟合的现象有所解决。特别是对比左上图与右下图的差距，可以看到样本数量对于过拟合问题是有影响的。

## 2、带惩罚项的解析解
首先根据式$(6)$我们需要确定最佳的超参数$\lambda$，因此我们通过PRML中提到的根均方(RMS)误差来确定，其中RMS的定义如式$(15)$。
$$ E_{RMS} = \sqrt{\dfrac{2E(\bold{w^*})}{N}} \tag{15}$$

在这里我们抽取了100次试验中的四次将实验结果放在了下图中，这四次中恰好有三次的最优的超参数均为$\lambda = e^{-7}$。

<center>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-8.png" width="45%">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-7.png" width="45%">
</figure>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-7(1).png" width="45%">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-7(2).png" width="45%">
</figure>
</center>

观察上面的四张图， 我们可以发现对于超参数$\lambda$的选择，在$(e^{-50}, e^{-30})$左右保持一种相对稳定的错误率；但是在$(e^{-30}, e^{-5})$错误率有一个明显的下降，所以下面在下面的完整100次实验中我们可以看到最佳参数的分布区间也大都在这个范围内；在大于$e^{-5}$的区间内，错误率有一个急剧的升高。

完整的100次实验的数据如下表格，我们可以看到最佳的超参数范围在$(e^{-9}, e^{-6})$之间，**因此在接下来的实验中我们将选取$\bold{\lambda = e^{-7}}$作为我们剩余中使用的最佳的超参数**。

<!-- 表格分割线 -->
|||||||||||||||
|-|-|-| ------ | -----|----|------|------|-----|-----|-|-|-|-|
|$ln\lambda$|-25 | -22  | -17 |-14| -13| -11 |-10 |-9| -8| -7| -6| -5| -3|
|$count$|1|1|2|4|2|5|6|11|26|29|11|1|1|
|
<!-- 分割线 -->
基于我们上面选择的超参数，我们进行对照实验，测试在多项式阶数为9，训练样本数量为20个的情况下进行对照实验，其中左上的图片是没有增加惩罚项的测试结果，其余三张为增加惩罚项后的测试结果。

<center>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_without_regulation/degree_9_number_training_20_number_test_100.png" width="45%">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-7_test_2.png" width="45%">
</figure>
<figure class="half">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-7_test_3.png" width="45%">
    <img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/analytical_solution_with_regulation/degree_9_number_20_hyper_-7_test_4.png" width="45%">
</figure>
</center>

我们可以看到增加正则项的后三张图片相比没有正则项的第一张图片，过拟合的现象得到了很好的解决，**这一结果验证了增加惩罚项对于过拟合的作用，以及经过上面实验得到的超参数$\lambda$是符合我们的要求的**。

## 3、优化解
在这里主要是利用梯度下降(Gradient descent)和共轭梯度法(Conjugate gradient)两种方法来求优化解。由于该问题是有解析解存在，并且在之前的实验报告部分已经列出，所以在此处直接利用上面的解析解进行对比。**此处使用的学习率固定为$rate = 0.01$，停止的精度要求为$1 \times 10 ^ {-6}$。**

对比实验主要分为2个变量的对比：**多项式的阶数**和**训练样本的数量**
。测试的结果的**迭代次数**全表如下：

<center>

|行号|多项式阶数|训练样本数量|梯度下降|共轭梯度|
|:-|:-:|:-:|:-:|:-:|
|1|3|10|47707|4|
|2|3|20|31010|4|
|3|3|50|19708|4|
|4|6|10|11368|5|
|5|6|20|8195|5|
|6|6|50|4209|7|
|7|9|10|28135|7|
|8|9|20|19348|7|
|9|9|20|2134|7|
</center>

首先在固定多项式阶数的情况下，**随着训练样本的增加，梯度下降的迭代次数均有所下降，但是对于共轭梯度迭代次数变化不大。**

其次在固定训练样本的情况下，**梯度下降迭代次数的变化，对于degree = 3的情况下多于degree = 9的情况，均多于degree = 6的情况。对于共轭梯度的而言，迭代次数仍然较少。**

总的来说，**对于梯度下降法，迭代次数多在10000次以上；而对于共轭梯度下降，则需要的迭代次数均不超过10次。**

下面给出这些实验的数据图，如下：
<center> 
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_3_number_train_10_gradient_47707_conjugate_4.png">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_3_number_train_20_gradient_31010_conjugate_4.png">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_3_number_train_50_gradient_19708_conjugate_4.png">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_6_number_train_10_gradient_11368_conjugate_5.png">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_6_number_train_20_gradient_8195_conjugate_5.png" width="95%">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_6_number_train_50_gradient_4209_conjugate_7.png" width="95%">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_9_number_train_10_gradient_28135_conjugate_7.png" width="95%">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_9_number_train_20_gradient_19348_conjugate_7.png" width="95%">
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/lab1/gradient/degree_9_number_train_50_gradient_2134_conjugate_7.png" width="95%">
</center>

综合以上几次实验的结果，**我们可以发现对于梯度下降法，其与解析解相比拟合的效果较差，而且迭代的次数比较多；而对于共轭梯度下降，在右边的图我们可以发现，其几乎与解析解表现相同，甚至不能明显的区分二者。**

# 五、结论
- **增加训练样本的数据可以有效的解决过拟合的问题**。
- 对于训练样本限制较多的问题，**通过增加惩罚项仍然可以有效解决过拟合问题**。
- 对于梯度下降法和共轭梯度法而言，**梯度下降收敛速度较慢，共轭梯度法的收敛速度快；且二者相对于解析解而言，共轭梯度法的拟合效果解析解的效果更好**。

# 六、参考文献
- [Pattern Recognition and Machine Learning.](https://www.springer.com/us/book/9780387310732)
- [Gradient descent wiki](https://en.wikipedia.org/wiki/Gradient_descent)
- [Conjugate gradient method wiki](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [Shewchuk J R. An introduction to the conjugate gradient method without the agonizing pain[J]. 1994](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf).

# 七、附录:源代码(带注释)

```python
import numpy as np
from matplotlib import pyplot as plt


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


def fitting(X_training, T_training):
    """ 求解析解 不带惩罚项 """
    w_analytical_with_regulation = np.dot(
        np.linalg.pinv(X_training), T_training)
    return w_analytical_with_regulation


def fitting_with_regulation(X_training, T_training, hyper=np.exp(-18)):
    """ 求解析解 带惩罚项 """
    X_T = X_training.transpose()
    # w_analytical_with_regulation = np.linalg.pinv(np.dot(X_T, X_training)
    # + np.eye(len(X_T)) * hyper) @ X_T @ T_training
    w_analytical_with_regulation = np.linalg.solve(
        np.eye(len(X_T)) * hyper + np.dot(X_T, X_training),
        np.dot(X_T, T_training))
    return w_analytical_with_regulation


def predict(X_Test, w):
    return np.dot(X_Test, w)


def E_rms(x, y):
    return np.sqrt(np.mean(np.square(x-y)))
    # 此处与PRML中相比差一个2^0.5的系数


def h(X_Train, T_training, hyper, number_train, w_0):
    """ 优化函数的导函数 """
    X_T = X_Train.transpose()
    return (X_T @ X_Train @ w_0 - X_T @ T_training + w_0 * np.exp(hyper))
    # return 1.0 / number_train * (X_T @ X_Train @ w_0 - X_T @ T_training
    # + w_0 * np.exp(hyper))
    # 此处如果增加number_train系数 会使迭代次数增多


def E(X_Train, T_training, hyper, number_train, w_0):
    """ 优化函数 """
    print(w_0)
    W_T = np.transpose([w_0])
    temp = X_Train @ w_0 - T_training
    # temp = np.linalg.norm(X_Train @ w_0 - T_training) ** 2
    temp = np.transpose(temp) @ temp
    return 0.5 / number_train * (temp + np.exp(hyper) * w_0 @ W_T)


def gradient_descent(X_Train, T_training, hyper, w_0, rate=0.01, delta=1e-6):
    """ 梯度下降法 
    Args:
        hyper:超参数，使用时以np.exp(hyper)为超参数
        rate:学习率
        delta:认为收敛的最小差距
    """
    loss = E(X_Train, T_training, hyper, len(X_Train), w_0)
    k = 0
    while True:
        w_gradient = w_0 - rate * \
            h(X_Train, T_training, hyper, len(X_Train), w_0)
        loss0 = E(X_Train, T_training, hyper, len(X_Train), w_gradient)
        if np.abs(loss0[0] - loss[0]) < delta:
            break
        else:
            print(k)
            k = k + 1
            print("abs:", np.abs(loss - loss0))
            print("loss:", loss)
            loss = loss0
            w_0 = w_gradient
    return w_gradient


def conjugate_gradient(X_Train, T_training, hyper, w_0, delta=1e-6):
    """ 共轭梯度法 """
    X_T = X_Train.transpose()
    b = X_T @ T_training
    A = X_T @ X_Train + np.identity(len(X_T)) * np.exp(hyper)
    r_0 = b - A @ w_0
    w_gradient = w_0
    p = r_0
    k = 0
    while True:
        print(k)
        k = k + 1
        alpha = np.linalg.norm(r_0) ** 2 / (np.transpose(p) @ A @ p)
        print("alpha:", alpha)
        w_gradient = w_gradient + alpha * p
        print("w_gradient:", w_gradient)
        r = r_0 - alpha * A @ p
        # r = b - A @ w_gradient
        print("r:", r)
        # q = np.linalg.norm(A @ w_gradient - b) / np.linalg.norm(b)
        if(np.linalg.norm(r) ** 2 < delta):
            break
        beta = np.linalg.norm(r)**2 / np.linalg.norm(r_0)**2
        print("beta:", beta)
        p = r + beta * p
        print("p:", p)
        r_0 = r
    return w_gradient


number_train = 50  # 训练样本的数量
number_test = 100  # 测试样本的数量
degree = 9  # 多项式的阶数
X_training, T_training = generateData(number_train)
X_test = np.linspace(0, 1, number_test)
X_Train = transform(X_training, degree=degree)
X_Test = transform(X_test, degree=degree)
Y = np.sin(2 * np.pi * X_test)

# 用于解析解(不带正则项)的实验
# title = "degree = " + str(degree) + ", number_train = " + str(number_train) + ", number_test = " + str(number_test)
# plt.title(title)
# plt.ylim(-1.5, 1.5)
# plt.scatter(X_training, T_training, facecolor="none",
#             edgecolor="b", label="training data")
# plt.plot(X_test, predict(X_Test, fitting(X_Train, T_training)), "r",
# label="analytical solution")
# plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
# plt.legend()
# plt.show()

# 用于解析解(带正则项)的实验 寻找最优的超参数
# 经过100次实验 最终得到的最优参数为e^-7
# anslist = []
# for i in range(100):
#     X_training, T_training = generateData(number_train)
#     X_test = np.linspace(0, 1, number_test)
#     X_Train = transform(X_training, degree=degree)
#     X_Test = transform(X_test, degree=degree)
#     Y = np.sin(2 * np.pi * X_test)
#     hyperTestList = []
#     hyperList = range(-50, 1)
#     for hyper in hyperList:
#         w_analytical_with_regulation = fitting_with_regulation(
#             X_Train, T_training, hyper=np.exp(hyper))
#         T_test = predict(transform(X_test, degree=degree),
#                         w_analytical_with_regulation)
#         # loss = Y - T_test
#         # ans = np.mean(loss @ np.transpose([loss]))
#         hyperTestList.append(E_rms(T_test, Y))
#     print(i)
#     bestHyper = hyperList[np.where(hyperTestList ==
#       np.min(hyperTestList))[0][0]]
#     print("bestHyper:", bestHyper, np.min(hyperTestList))
#     anslist.append(bestHyper)
# myset = set(anslist)
# for item in myset:
#     print("the %d has found %d" %(item,anslist.count(item)))
# title = "degree:" + str(degree) + ",number_train:" + str(number_train)
# annotate = "$\lambda = e^{" + str(bestHyper) + "}$"
# plt.title(title)
# plt.ylabel("$E_{RMS}$")
# plt.xlabel("$ln \lambda$")
# plt.annotate(annotate, xy=(-30, 0.3))
# plt.plot(hyperList, hyperTestList, 'o-', mfc="none", mec="b", ms=5,
#  label="Test")
# plt.legend()
# plt.show()

# 此处用于确认带有惩罚项的解析解的正确性实验
# bestHyper = -7 #此处的最佳的超参数是经过上面提到的实验中确定的
# w_analytical_with_regulation = fitting_with_regulation(
#     X_Train, T_training, hyper=np.exp(bestHyper))
# T_test = predict(X_Test, w_analytical_with_regulation)
# title = "degree = " + str(degree) + ", number_train = " + str(number_train) + ", number_test = " + str(number_test)
# annotate = "$\lambda = e^{" + str(bestHyper) + "}$"
# plt.title(title)
# plt.ylim(-1.5, 1.5)
# plt.scatter(X_training, T_training, facecolor="none",
#             edgecolor="b", label="training data")
# plt.plot(X_test, T_test, "r", label="analytical with regulation")
# plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
# plt.annotate(annotate, xy=(0.3, -0.5))
# plt.legend()
# plt.show()


bestHyper = -7  # 此处的最佳的超参数是经过上面提到的实验中确定的
w_analytical_with_regulation = fitting_with_regulation(
    X_Train, T_training, hyper=np.exp(bestHyper))
T_test = predict(X_Test, w_analytical_with_regulation)
w_0 = np.zeros(degree+1)
w_gradient = gradient_descent(X_Train, T_training, bestHyper, w_0)
w_conjugate = conjugate_gradient(X_Train, T_training, bestHyper, w_0)
print()
print("w_analytical_with_regulation(Analytical solution):\n",
      w_analytical_with_regulation)
print("w_gradient(Gradient descent):\n", w_gradient)
print("w_conjugate(Conjugate gradient):\n", w_conjugate)

title = "degree = " + str(degree) + ", number_train = " + \
    str(number_train) + ", number_test = " + str(number_test)
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title(title)
plt.ylim(-1.5, 1.5)
plt.scatter(X_training, T_training, facecolor="none",
            edgecolor="b", label="training data")
plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
plt.plot(X_test, T_test, "r", label="Analytical with regulation")
plt.plot(X_test, predict(X_Test, w_gradient), "c", label="Gradient descent")
plt.legend()

plt.subplot(122)
plt.ylim(-1.5, 1.5)
plt.title(title)
plt.scatter(X_training, T_training, facecolor="none",
            edgecolor="b", label="training data")
plt.plot(X_test, Y, "g", label="$\sin(2\pi x)$")
plt.plot(X_test, T_test, "r", label="Analytical regulation")
plt.plot(X_test, predict(X_Test, w_conjugate), "m",
         label="Conjugate gradient")
plt.legend()
plt.show()
```