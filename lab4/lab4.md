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
实验题目：PCA模型实验 
</font></center>
<br/>
<br/>
<center> <font size = 4> 学号：1160300314 </font></center>
<center> <font size = 4> 姓名：朱明彦 </font></center>

<div STYLE="page-break-after: always;"></div>
<!-- 此处用于换行 -->

# 一、实验目的
实现一个PCA模型，能够对给定数据进行降维(即找到其中的主成分)，**可以利用已有的矩阵特征向量提取方法。**
# 二、实验要求及实验环境

## 实验要求
**测试**
1. 首先人工生成一些数据（如三维数据），让它们主要分布在低维空间中，如首先让某个维度的方差远小于其它维度，然后对这些数据旋转。生成这些数据后，用你的PCA方法进行主成分提取。
2. 利用手写体数字数据mnist，用你实现PCA方法对该数据降维，找出一些主成分，然后用这些主成分对每一副图像进行重建，比较一些它们与原图像有多大差别（可以用信噪比衡量）。
## 实验环境
- **OS**: Ubuntu 16.04.5 LTS
- python 3.7.0
# 三、设计思想(本程序中用到的主要算法及数据结构)

## 1.算法原理
<!-- TODO -->
## 2.算法的实现
给定样本集${\bf D}=\{{\bf x}_1, {\bf x}_2, \dots, {\bf x}_m\}$和低维空间的维数$d$
1. 对所有的样本进行中心化操作：
    1. 计算样本均值${\bf \mu} = \frac{1}{m}\sum\limits_{j=1}^m{\bf x}_j$
    2. 所有样本减去均值${\bf x}_j = {\bf x}_j - {\bf \mu},\ j \in \{1, 2, \dots, m\}$
2. 计算样本的协方差矩阵${\bf X}^T{\bf X}$
3. 对协方差矩阵${\bf X}^T{\bf X}$进行特征值分解
4. 取最大的$d$个特征值对应的单位特征向量${\bf w}_1, {\bf w}_2, \dots, {\bf w}_d$，构造投影矩阵${\bf W} = ({\bf w}_1, {\bf w}_2, \dots, {\bf w}_d)$
5. 输出投影矩阵$\bf W$与样本均值$\bf \mu$
# 四、实验结果分析
## 1.生成数据的测试
**为了方便进行数据可视化，在这里只进行了2维数据和3维数据的在PCA前后的对比实验。**
### 2维数据的测试
在2维数据的测试中，选择使用2维高斯分布产生样本，使用的参数为：
$$
{\bf mean = }
\left[
\begin{matrix}
    -2, 2
\end{matrix} 
\right], \ 
{\bf cov = }
\left[
\begin{matrix}
        0.01 & 0\\
        0 & 1\\
\end{matrix}
\right]
$$
可以看到第1维的方差远小于第2维的方差$(0.01 \ll 1)$，因此有直观感觉在第2维包含了更多的信息，所以直接进行PCA，得到的结果如下：
<center>
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/ML_lab4/2_dimension.png">
</center>
可以看到在PCA之后的数据分布在直线(1维)上，另外其在横轴上的方差更大，纵轴上的方差更小(注意横轴纵轴在单位长度上表示的大小不同)，所以在进行PCA之后得到的直线与横轴接近。

### 3维数据的测试
在3维数据的测试中，使用3维高斯分布随机产生样本，使用的参数为：
$$
{\bf mean = }
\left[
\begin{matrix}
    1, 2, 3
\end{matrix} 
\right], \ 
{\bf cov = }
\left[
\begin{matrix}
        0.01 & 0 & 0\\
        0 & 1 & 0 \\
        0 & 0 & 1
\end{matrix}
\right]
$$
同样，可以看到第1维的方差是远小于其余两个维度的，所以在第1维相较于其他两维信息更少，进行PCA得到的结果如下：
<center>
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/ML_lab4/3_dimension_origin.png">
</center>
可以看到在底面的一个轴的单位长度表示的长度更小，即在原始数据上的第1维数据，对上面的图片进行旋转，我们可以看到：
<center>
<img src="https://raw.githubusercontent.com/1160300314/Figure-for-Markdown/master/ML_lab4/3_dimension_plat.png">
</center>
降维后的数据

## 2.[mnist]((http://yann.lecun.com/exdb/mnist/))手写数据集测试
# 五、结论

# 六、参考文献
- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [Christopher Bishop. Pattern Recognition and Machine Learning.](https://www.springer.com/us/book/9780387310732)
- [周志华 著. 机器学习, 北京: 清华大学出版社, 2016.1](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm)
- [AI算法工程师手册 数据降维](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/chapters/10_PCA.html)
# 七、附录:源代码(带注释)
仅有`lab4.py`文件，见压缩包