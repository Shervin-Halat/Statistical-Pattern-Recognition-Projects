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

cov1 = np.cov(sample1.T)
cov2 = np.cov(sample2.T)


plt.scatter(sample1[:,0],sample1[:,1], s = 0.5,label = 'Class 1')
plt.scatter(sample2[:,0],sample2[:,1], s = 0.5,label = 'Class 2')


plt.xlabel('X1')
plt.ylabel('X2')
plt.title('projected classes samples on PCA line')

sampletotal = np.concatenate((sample2, sample1) , axis = 0)
samplemean = sampletotal.mean(axis = 0)
sampletotal2 = sampletotal-samplemean

'''
plt.scatter(sampletotal2[:,0],sampletotal2[:,1],s = 0.5 ,label = 'shifted data by mean vector')
'''

samplecov = np.cov(sampletotal2.T)
w,v = eig(samplecov)
eigvec = v[:,w.argmax()]

print(eig(samplecov))

val1 = (eigvec.dot(sample1.T))/((np.linalg.norm(eigvec))**2)
sampleproj1 = np.array([eigvec*i for i in val1])
plt.scatter(sampleproj1[:,0],sampleproj1[:,1] , s = 10 ,c = 'tab:blue', label= 'class1')

val2 = (eigvec.dot(sample2.T))/((np.linalg.norm(eigvec))**2)
sampleproj2 = np.array([eigvec*i for i in val2])
plt.scatter(sampleproj2[:,0],sampleproj2[:,1] , s = 6 , c = 'tab:orange' , label= 'class2')

xaxis = np.linspace(-2,10,3)
yaxis = xaxis*eigvec[1]/eigvec[0]
plt.plot(xaxis, yaxis, label = 'PCA' , c='red' , linewidth=1)

plt.legend()
plt.xlim(-20,20)
plt.ylim(0,40)
#plt.axis('equal')
plt.show()
