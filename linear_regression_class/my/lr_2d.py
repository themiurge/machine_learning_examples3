import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import lr

# load the data
X = []
Y = []
for line in open('../data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

# caluculate weights
w = lr.linreg_multi(X, Y)
print(w)

# calculate estimate
Yhat = np.dot(X, w)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,2], Yhat)
plt.show()

# calculate r-squared
print("r-squared:", lr.r_squared(Y, Yhat))
