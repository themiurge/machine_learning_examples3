# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
N = 50
X = np.linspace(0, 10, N)
Y = .5*X + np.random.randn(N)
plt.scatter(X,Y)
plt.show()
Y[-1] += 30
Y[-2] += 30
plt.scatter(X,Y)
plt.show()
X = np.vstack([np.ones(N), X]).T
X
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_hat = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_hat)
plt.show()
l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_hatl2 = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_hat, label='maximum likelihood')
plt.plot(X[:,1], Y_hatl2, label='map')
plt.legend()
plt.show()
