import numpy.random as rd
import numpy as np
from metriclearning import *
import matplotlib.pyplot as plt

X=rd.random((200,100))
X[0:100,10]=1; y=np.ones((1,200))
X[100:,10]=0; y[0,100:]=0

print(y)
gamma=suvrel(X,y[0])
print('X: ')
print(X)
print('gamma: ')
print(gamma)
plt.plot(gamma)
plt.show()
print('newX:')
newx = [gamma*x for x in X]
print(newx)
plt.plot(np.transpose(newx))
plt.show()
