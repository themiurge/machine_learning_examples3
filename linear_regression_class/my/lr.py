import numpy as np

# function that computes linear regression coefficients (1-D)
def linreg(X, Y):
    xsq = X.dot(X)
    xm = X.mean()
    xs = X.sum()
    xys = X.dot(Y)
    ym = Y.mean()
    denominator = xsq - xm * xs
    a = (xys - ym * xs) / denominator
    b = (ym * xsq - xm * xys) / denominator
    return a, b

# function that computes linear regression coefficients (multi-D)
def linreg_multi(X, Y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

#function that computes regression error
def r_squared(Y, Yhat):
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    return 1 - d1.dot(d1) / d2.dot(d2)
