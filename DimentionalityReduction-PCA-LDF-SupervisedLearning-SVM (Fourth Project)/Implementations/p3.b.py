import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from numpy.linalg import eig

mean1 = np.array([10,10])
mean2 = np.array([10,22])
cov = np.array([[9,4],[4,4]])

sample1 = multivariate_normal(mean1, cov, 1000)
#print(sample1)
sample2 = multivariate_normal(mean2, cov, 1000)
#print(sample2)

print('%%', sample1.shape)

cov1 = np.cov(sample1.T)
cov2 = np.cov(sample2.T)

plt.scatter(sample1[:,0],sample1[:,1], s = 0.5,label = 'Class 1')
plt.scatter(sample2[:,0],sample2[:,1], s = 0.5,label = 'Class 2')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('normal distributed classes samples')

sampletotal = np.concatenate((sample2, sample1) , axis = 0)
samplemean = sampletotal.mean(axis = 0)
sampletotal2 = sampletotal-samplemean
plt.scatter(sampletotal2[:,0],sampletotal2[:,1],s = 0.5 ,label = 'shifted data by mean vector')

samplecov = np.cov(sampletotal2.T)
w,v = eig(samplecov)
eigvec = v[:,w.argmax()]

print(eig(samplecov))

'''
val = (eigvec.dot(sampletotal2.T))/((np.linalg.norm(eigvec))**2)
sampleproj = np.array([eigvec*i for i in val])
plt.scatter(sampleproj[:,0],sampleproj[:,1])
'''

xaxis = np.linspace(-2,2,3)
yaxis = xaxis*eigvec[1]/eigvec[0]
plt.plot(xaxis, yaxis, label = 'PCA' , c='red')

plt.legend()
plt.axis('equal')
plt.show()
