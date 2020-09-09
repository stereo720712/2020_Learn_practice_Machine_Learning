#predict function :sigmod

import  numpy as np
from  math import  exp
def sigmod(X, theta):
    return  1 / (1 + exp(theta*X))


def cost_reg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmod(X, theta)))
    second = np.multiply((1-y), np.log(1- sigmod(X,theta)))
    reg = (learningRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2)))
    return np.sum(first - second) / len(X)


