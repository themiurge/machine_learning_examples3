import re
import numpy as np
import matplotlib.pyplot as plt
import lr

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('../moore.csv'):
    r = line.split('\t')
    if len(r) < 3:
        print("bad line: {0}".format(line))
    else:
        x = int(non_decimal.sub('', r[2].split('[')[0]))
        y = int(non_decimal.sub('', r[1].split('[')[0]))
        X.append(x)
        Y.append(y)

X = np.array(X)
Y = np.array(Y)

# plot data - linear
plt.scatter(X, Y)
plt.show()

# plot data - log
Ylog = np.log(Y)
plt.scatter(X, Ylog)
plt.show()

# get linear regression coefficients
a, b = lr.linreg(X, Ylog)

# get Yhat
Yhat = a * X + b

# plot it
plt.scatter(X, Ylog)
plt.plot(X, Yhat)
plt.show()

# get r-squared
print ("r-squared is", lr.r_squared(Ylog, Yhat))

# see how many years it takes for the transistor count to double
years = np.log(2) / a
print ("it takes", years, "years for the transistor count to double")
