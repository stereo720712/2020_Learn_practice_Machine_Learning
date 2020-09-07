import numpy as np

m = 20

X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

y = np.array([3,4,5,5,2,4,7,8,11,8,
              12,11,13,13,16,17,18,17,19,21]).reshape(m, 1)

alpha = 0.01

def error_function(theta, X, y):
    '''
        Loss function
    '''
    diff = np.dot(X, theta) - y
    # matrix square is a * aT
    loss = (1./2*m) * np.dot(np.transpose(diff), diff)
    return loss


def gradient_function(theta, X, y):
    '''theta gradient'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

# batch --> degree
# randrom gradient decrease
def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape((2, 1))
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        print("---running---")
        theta = theta - alpha * gradient
        print("new theta: ", theta)
        gradient = gradient_function(theta, X, y)
        print("new gradient: ", gradient)
        print('new error: ', error_function(theta, X, y)[0, 0])
    return theta

optimal = gradient_descent(X, y , alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0, 0])
