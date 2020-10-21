from sklearn import datasets
from sklearn import svm
import numpy as np
np.random.seed(0)
iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

#  使用线性核 SVC分类支持向量机 SVR回归支持向量机
clf = svm.SVC(kernel='linear')
clf.fit(iris_x_train,iris_y_train)

iris_y_predict = clf.predict(iris_x_test)

score = clf.score(iris_x_test,iris_y_test,sample_weight=None)
print('iris_y_predict = ')
print(iris_y_predict)
print('iris_y_test = ')
print(iris_y_test)
print('Accuracy:',score)
