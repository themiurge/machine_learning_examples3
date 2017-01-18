import numpy as np
import matplotlib.pyplot as plt
import lr

#inport data
X = []
Y = []
for line in open('../data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

# plot data
plt.scatter(X[:,1], Y)
plt.show()

# calculate weights - this is multiple linear regression
# the only difference is in how we create the input X
w = lr.linreg_multi(X, Y)
Yhat = np.dot(X, w)

# plot it
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()

# calculate r-squared
print("r-squared:", lr.r_squared(Y, Yhat))
