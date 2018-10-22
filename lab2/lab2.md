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
实验题目：logistics Regression
</font></center>
<br/>
<br/>
<center> <font size = 4> 学号：1160300314 </font></center>
<center> <font size = 4> 姓名：朱明彦 </font></center>

<div STYLE="page-break-after: always;"></div>
<!-- 此处用于换行 -->

# 一、实验目的
- 理解逻辑回归模型
- 掌握逻辑回归模型的参数估计法
# 二、实验要求及实验环境
## 实验要求
- 实现两种损失函数的参数估计（1、无惩罚项；2、加入对参数的惩罚），可以采用梯度下降、共轭梯度或者牛顿法等。

### 验证
1. 可以手工生成两个分别类数据（可以用高斯分布），验证你的算法。考察类条件分布不满足朴素贝叶斯假设，会得到什么样的结果。
2. 逻辑回归有广泛的用处，例如广告预测。可以到UCI的网站上，找一实际数据加以测试。
## 实验环境
- **OS**: Ubuntu 16.04.5 LTS
- python 3.7.0
# 三、设计思想(本程序中的用到的主要算法及数据结构)

## 1.算法原理
<!-- TODO Logistics Regression原理描述 可以参考西瓜书相应章节-->
Logistic回归的**基本思想**就是利用朴素贝叶斯的假设取计算$\mathsf{P}(Y|X)$:即利用$\mathsf{P}(Y)$，$\mathsf{P}(X|Y)$以及各个维度之间计算条件独立的假设来计算$\mathsf{P}(Y|X)$。

考虑二分类问题，$f:X \rightarrow Y$，其中$X$为实数向量, $X = <X_1, X_2, \dots, X_n>$，$Y \in \{0, 1\}$，且有对于所有的$X_i$在给定$Y$的前提下均有条件独立(即各维条件独立)成立；并且有$\mathsf{P}(X_i|Y = y_k) \sim N(\mu_{ik}, \sigma_i)$，$\mathsf{P}(Y) \sim B(\pi)$成立。

那么我们求解$\mathsf{P}(Y|X)$的方式，就可以有下面这种方式的推导：

$$ \mathsf{P}(Y = 0|X) = \cfrac{P(Y = 0) P(X|Y=0)}{P(X)} \tag{1}$$

将式$(1)$下面全概率公式展开可以得到：

$$ \mathsf{P}(Y = 0|X) = \cfrac{P(Y = 0) P(X|Y=0)}{P(Y = 1)P(X|Y=1) + P(Y = 0)P(X|Y=0)} \tag{2}$$

将$(2)$式右边，上下同时除以$P(Y=0)P(X|Y=0)$，得到：

$$ P(Y = 0 | X) = \cfrac{1}{1 + \cfrac{P(Y=1)P(X|Y=1)}{P(Y=0)P(X|Y=0)}} = \cfrac{1}{1 + exp\left(ln\cfrac{P(Y=1)P(X|Y=1)}{P(Y=0)P(X|Y=0)}\right)} \tag{3}$$

又由于$Y$符合伯努利分布，我们可以将$\pi = \widehat{P}(Y = 1)$带入$(3)$中得到:

$$ P(Y=0|X) = \cfrac{1}{1 + exp\left(ln(\dfrac{\pi}{1-\pi}) + ln\cfrac{P(X|Y=1)}{P(X|Y=0)}\right)} \tag{4}$$

因为有着朴素贝叶斯的假设，我们可以讲向量各维的分布展开：

$$ P(Y=0|X) = \cfrac{1}{1 + exp\left(ln(\dfrac{\pi}{1-\pi}) + \sum_{i}\left( ln\cfrac{P(X_i|Y=1)}{P(X_i|Y=0)} \right) \right)} \tag{5}$$

另由于各个维度使符合高斯分布的，我们可以将各个维度的高斯分布函数$P(X_i|Y=y_k)= \cfrac{1}{\sigma_{ik}\sqrt{2\pi}} \ exp(\cfrac{-(x-\sigma_{ik})^2}{2\sigma_{ik}^2})$带入，可以有:

$$ P(Y=0|X) = \cfrac{1}{1 + exp\left(ln(\dfrac{\pi}{1-\pi}) + \sum_{i}\left( \cfrac{\mu_{i1} - \mu_{i0}}{\sigma_i^2}X_i + \cfrac{\mu_{i0}^2 - \mu_{i1}^2}{2\sigma_i^2} \right) \right)} \tag{6}$$

将其写为向量的形式，可以转化为：

$$ P(Y=0|X) = \cfrac{1}{1 + exp(\mathbf{w^TX})} \tag{7}$$

其中，$\bold{w}_0= \sum_i^n\left(\cfrac{\mu_{i0}^2 - \mu_{i1}^2}{2\sigma_i^2}\right) + ln(\cfrac{\pi}{1-\pi})$， $\bold{w}_i = \cfrac{\mu_{i1} - \mu_{i0}}{\sigma_i^2},\ i> 0$, $
\bold{X} = 
\left[
\begin{matrix}
   1 \\ X_1 \\ \vdots \\ X_n
\end{matrix}
\right]$

利用归一化的特性，我们可以得到:
$$ P(Y=1|X) = 1 - \cfrac{1}{1 + exp(\mathbf{w^TX})} = \cfrac{exp(\mathbf{w^TX})}{1 + exp(\mathbf{w^TX})}\tag{8}$$

当前我们有(或者有生成的数据) $\{<X^1, Y^1>, \dots, <X^l, Y^l>\}$，我们通过最大条件似然法对参数$\bold{w}$进行估计：
$$ \bold{w}_{MCLE} = \mathop{arg} \mathop{max}\limits_{\bold{w}}\prod\limits_{l}P(Y^l|X^l, \bold{w}) \tag{9}$$

对式$(9)$两边同时取对数，我们可以有：
$$ l(\bold{w}) \equiv ln \prod\limits_{l}P(Y^l|X^l,\bold{w}) = \sum\limits_{l}P(Y^l|X^l,\bold{w}) \tag{10}$$

将式$(8)(9)$带入$(10)$中，我们得到：

$$ l(\bold{w}) = \sum\limits_{l}\left(Y^l\bold{w^TX} - ln(1 + exp(\bold{w^TX}))\right) \tag{11}$$

最大化式$(11)$就等价于最小化式$(12)$:

$$ \mathcal{L}(\bold{w}) = \sum\limits_{l}\left(-Y^l\bold{w^TX} + ln(1 + exp(\bold{w^TX}))\right) \tag{12}$$

式$(12)$是关于$\bold{w}$的高阶可导连续凸函数[2]，根据凸优化的理论，在这里我们可以用梯度下降法、牛顿法等求解其最优解，在算法实现方面详述。

为了避免过拟合现象，我们仿照lab1的经验，对于式$(12)$增加惩罚项，其中$\lambda$为超参数：

$$ \mathcal{L}(\bold{w}) = \frac{\lambda}{2}\bold{w^Tw} + \sum\limits_{l}\left(-Y^l\bold{w^TX} + ln(1 + exp(\bold{w^TX}))\right) \tag{13}$$
## 2.算法的实现
<!-- TODO logistics Regression实现的两种方式 利用梯度下降和牛顿法进行优化 -->
算法实现部分，此处选择使用梯度下降法实现以及使用牛顿法进行优化。
### 2.1 梯度下降实现
根据Lab1的经验，其实对于梯度下降法的使用没有什么变化，只是将优化的函数做了一下修改，所以我们可以得到其一阶导数，然后在$t+1$轮得到的迭代式子如下，其中$\alpha$为学习率:
$$ \bold{w^{t+1}} = \bold{w^t} - \alpha\cfrac{\partial \mathcal{L}}{\partial \bold{w}}(\bold{w^t})$$

$$ \cfrac{\partial \mathcal{L}}{\partial \bold{w}} = - \sum\limits_{i=1}^lX_i\left(Y_i - \cfrac{exp(\bold{w^TX})}{1+exp(\bold{w^TX})}\right)$$
这种直接将式$(13)$求导进行迭代的方式，存在在数据特别多(即$l$特别大)的情况下，有可能导致上溢出发生，基于此，我们将式$(13)$归一化，防止其溢出，得到式$(14)$:

$$ \mathcal{L}(\bold{w}) = \frac{\lambda}{2l}\bold{w^Tw} + \frac{1}{l} \sum\limits_{l}\left(-Y^l\bold{w^TX} + ln(1 + exp(\bold{w^TX}))\right) \tag{14}$$

然后再进行迭代，就可以避免上溢出的现象。
### 2.2 牛顿法实现
与梯度下降法实现类似，此处我们有在$t+1$轮迭代的式子如下:
$$ \bold{w^{t+1}} = \bold{w^t} - \left(\cfrac{\partial^2\mathcal{L}}{\partial \bold{w} \partial \bold{w^T}}\right)^{-1}\cfrac{\partial \mathcal{L}}{\partial \bold{w}}$$

$$ \cfrac{\partial^2\mathcal{L}}{\partial \bold{w} \partial \bold{w^T}} = \sum\limits_{i=1}^l\left(X_iX_i^T\ \cfrac{exp(\bold{w^TX})}{1+exp(\bold{w^TX})}\ \cfrac{1}{1+exp(\bold{w^TX})}\right)$$
# 四、实验结果分析
<!-- 实验结果分析 主要在于对比实验和UCI数据的实验 对比实验更改不同的均值与方差
以及破坏Naive Bayes的条件 对实验的影响 -->
<!-- 3种破坏方式，
1.破坏方差仅与类别相关，即，使方差与类别和维度均有关系 
2.破坏各位之间的条件独立性，即，各维之间的协方差矩阵不为对角阵
1. 1与2均将其破坏-->
# 五、结论

# 六、参考文献
- [Christopher Bishop. Pattern Recognition and Machine Learning.](https://www.springer.com/us/book/9780387310732)

- [周志华 著. 机器学习, 北京: 清华大学出版社, 2016.1.](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm)
- [Newton's Method, wiki](https://en.wikipedia.org/wiki/Newton%27s_method)
# 七、附录:源代码(带注释)

```python
print("Hello world")

import numpy as np

np.one(10)
```