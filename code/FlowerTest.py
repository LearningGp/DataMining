from sklearn import datasets  #sklearn的数据集

from sklearn.neighbors import KNeighborsClassifier #sklearn模块的KNN类

import numpy as np #矩阵运算库numpy

np.random.seed(0) #设置随机种子，不设置的话默认是按系统时间作为参数，设置后可以保证我们每次产生的随机数是一样的

iris=datasets.load_iris() #获取鸢尾花数据集

iris_x=iris.data  #数据部分

iris_y=iris.target  #类别部分

#从150条数据中选140条作为训练集，10条作为测试集。permutation 接收一个数作为参数(这里为数据集长度150),产生一个0-149乱序一维数组

randomarr= np.random.permutation(len(iris_x))

iris_x_train = iris_x[randomarr[:-10]]  #训练集数据

iris_y_train = iris_y[randomarr[:-10]]  #训练集标签

iris_x_test = iris_x[randomarr[-10:]]  #测试集数据

iris_y_test = iris_y[randomarr[-10:]]  #测试集标签

#定义一个knn分类器对象

knn = KNeighborsClassifier()

#调用该对象的训练方法，主要接收两个参数：训练数据集及其类别标签

knn.fit(iris_x_train, iris_y_train) 

#调用预测方法，主要接收一个参数：测试数据集

iris_y_predict = knn.predict(iris_x_test)

#计算各测试样本预测的概率值 这里我们没有用概率值，但是在实际工作中可能会参考概率值来进行最后结果的筛选，而不是直接使用给出的预测标签

probility=knn.predict_proba(iris_x_test) 

#计算与最后一个测试样本距离最近的5个点，返回的是这些样本的序号组成的数组

neighborpoint=knn.kneighbors([iris_x_test[-1]],5)

#调用该对象的打分方法，计算出准确率

score=knn.score(iris_x_test,iris_y_test,sample_weight=None)

#输出测试的结果

print('iris_y_predict = ') 

print(iris_y_predict) 

#输出原始测试数据集的正确标签，以方便对比

print('iris_y_test = ')

print(iris_y_test) 
#输出准确率计算结果
print('Accuracy:',score) 
