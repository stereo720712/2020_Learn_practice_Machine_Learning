import numpy as np

m = 20

X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

y = np.array([3,4,5,5,2,4,7,8,11,8,
              12,11,13,13,16,17,18,17,19,21]).reshape(m, 1)

alpha = 0.1

def error_function(theta, X, y):
    '''
        Loss function
    '''
    diff = np.dot(X, theta) - y
    # matrix square is a * aT
    return (1./2*m) * np.dot(np.transpose(diff),diff)

def gradient_function(theta, X, y):
    '''theta gradient'''
    diff = np.dot(X, theta) - y

