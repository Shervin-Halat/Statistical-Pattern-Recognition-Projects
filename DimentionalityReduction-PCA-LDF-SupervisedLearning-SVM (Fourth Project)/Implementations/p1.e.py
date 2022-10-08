import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

mean1 = np.array([10,10])
mean2 = np.array([10,22])
cov = np.array([[9,4],[4,4]])

sample1 = multivariate_normal(mean1, cov, 1000)
sample2 = multivariate_normal(mean2, cov, 1000)

plt.scatter(sample1[:,0],sample1[:,1], 0.5)
plt.label('Class1')
plt.scatter(sample2[:,0],sample2[:,1], 0.5)
plt.label('Class2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('')
plt.legend()
plt.show()
