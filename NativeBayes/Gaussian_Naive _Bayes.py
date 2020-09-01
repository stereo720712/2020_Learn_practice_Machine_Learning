# https://scikit-learn.org/stable/modules/naive_bayes.html

from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import GaussianNB

x, y = load_iris(return_X_y = True)

x_train, x_test, y_train , y_test = train_test_split(x, y , test_size= 0.5, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
print("Number of mislabeled points out of a total %d points: %d" % (x_test.shape[0],(y_test != y_pred).sum()))

