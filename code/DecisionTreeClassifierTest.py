from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
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

clf = DecisionTreeClassifier(max_depth=4)
#  训练
clf.fit(iris_x_train,iris_y_train)

from IPython.display import Image
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(clf,out_file=None,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('graph.png')

iris_y_predict = clf.predict(iris_x_test)
score=clf.score(iris_x_test,iris_y_test,sample_weight=None)
print('iris_y_predict = ')
print(iris_y_predict)
print('iris_y_test = ')
print(iris_y_test)
print('Accuracy:',score)
