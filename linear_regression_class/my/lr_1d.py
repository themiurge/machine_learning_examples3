import numpy as np
import matplotlib.pyplot as plt
import lr

# load the data
X = []
Y = []
for line in open('../data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# compute linear regression coefficients
a, b = lr.linreg(X, Y)

# compute Yhat
Yhat = a * X + b

# print coefficients
print ("plotting Y = {0}X + {1}".format(a, b))

# plot the data
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# print value of r-squared
print ("r-squared is", lr.r_squared(Y, Yhat))
