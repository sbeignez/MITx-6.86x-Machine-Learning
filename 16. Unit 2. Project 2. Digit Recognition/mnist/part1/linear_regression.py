import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    n , d = X.shape
    A1 = np.linalg.inv(np.dot(X.T,X) + lambda_factor * np.identity(d))
    A2 = np.dot(X.T, Y)
    theta = np.dot(A1,A2)

    return theta

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)


# X = np.array(range(12)).reshape(4,3)
# Y = np.array(range(4)).reshape(4,)
# print(X)
# print(np.dot(X.T,X))
# I = np.identity(3)
# print(np.dot(X.T,X)+ I)

# print(closed_form(X, Y, 2))
