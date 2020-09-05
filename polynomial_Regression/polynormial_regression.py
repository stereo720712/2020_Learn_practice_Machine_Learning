import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def runplt():
    plt.figure()
    plt.title(u'diameter-cost curver')
    plt.xlabel(u'diameter')
    plt.ylabel(u'cost')
    plt.axis([0,25,0,25])
    plt.grid(True)
    return plt
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
xx_data = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx_data)
plt = runplt()
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)
# plt.show()

# ex1
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic),'r-')
plt.show()
print(X_train)
print('\n')
print(X_train_quadratic)
print('\n')
print(X_test)
print('\n')
print(X_test_quadratic)
print('\n')
print('1 r-squared ', regressor.score(X_test,y_test))
print('\n')
print('2 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))

#
