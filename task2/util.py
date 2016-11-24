"""
Function to know the sparsity of the data and know if it is necessary to use and adaptative gradient
descent method as AdaGrad.
"""

import numpy as np

with open('data/handout_train.txt', 'r') as f:
    lines = f.readlines()

X = []
Y = []

for line in lines:
    l = line.strip().split(' ')
    Y.append(int(l[0]))
    x = list(map(float, l[1:]))
    X.append(x)

X = np.array(X)
Y = np.array(Y)

non_zero = float(np.sum(X!=0.)) / X.size * 100.
zero = float(np.sum(X==0.)) / X.size * 100.

print('Sparsity: {:.2f}%'.format(zero))
print('Density: {:.2f}%'.format(non_zero))
