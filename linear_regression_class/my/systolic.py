import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lr

df = pd.read_excel('../../data/mlr02.xls')

X = df.as_matrix()

plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

df['ones'] = 1
df['noise'] = np.random.randn(len(df['X1']))
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
Xnoise = df[['X2', 'X3', 'ones', 'noise']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = lr.linreg_multi(X, Y)
    Yhat = X.dot(w)
    return lr.r_squared(Y, Yhat)

print("r-squared X2:", get_r2(X2only, Y))

print("r-squared X3:", get_r2(X3only, Y))

print("r-squared X2, X3:", get_r2(X, Y))

print("r-squared X2, X3, noise:", get_r2(Xnoise, Y))
