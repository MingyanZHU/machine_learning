import numpy as np
import pandas as pd

dataset = pd.read_csv("./mushrooms.csv")
print(dataset.head())

print(dataset.isnull().sum())
print(dataset['odor'])

x = dataset.drop('class', axis=1)
y = dataset['class']
print(x.head())
print(y.head())
# houseprice['MSZoning']=houseprice['MSZoning'].map({'RL':1,'RM':2,'RR':3,}).astype(int)
# 将MSZoning中的字符串变成对应的数字表示
# x['cap-shape'] = x['cap-shape'].map({''})
# print(type(x))
x = pd.get_dummies(x)
print(x.head())

from sklearn import preprocessing
xnor = preprocessing.normalize(x)
xscale = preprocessing.scale(x)
print(xnor.shape)
print(xscale)

# TODO 
# 将mushrooms.csv的数据提取出来 (如果不允许使用sklearn，规格化可能需要自己写)
# sklearn的源码，查看classification的结果写法
# 测试集与训练集分开
# 生成数据的问题    