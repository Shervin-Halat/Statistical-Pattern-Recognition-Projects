import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from numpy.linalg import eig
from numpy.linalg import norm
from numpy.linalg import inv

x1 = np.array([[3,4],[1,2],[4,3],[2,2],[5,2]])
x2 = np.array([[8,7],[5,9],[7,6],[9,4],[6,6]])
'''
plt.scatter(x1[:,0],x1[:,1],label = 'X1')
plt.scatter(x2[:,0],x2[:,1],label = 'X2')
plt.legend()
'''
ax = plt.gca()
ax.set_ylim([0,10])
ax.set_xlim([0,10])
#plt.show()

m1 = np.mean(x1, axis=0)
m2 = np.mean(x2, axis=0)
c1 = np.cov(x1.T)
c2 = np.cov(x2.T)

sw = np.linalg.inv(c1 + c2)
v = sw.dot(m1 - m2)
print(v.shape)
print(v)

xaxis = np.linspace(-2,10,3)
yaxis = xaxis*v[1]/v[0]
plt.plot(xaxis, yaxis, label = 'LDA' , c='red' , linewidth=1)


x1p = (v.dot(x1.T))/np.linalg.norm(v)
print(x1p,x1p.shape)
x2p = (v.dot(x2.T))/np.linalg.norm(v)

mp1 = np.mean(x1p)
mp2 = np.mean(x2p)
sp1 = 4*np.var(x1p)
sp2 = 4*np.var(x2p)

print('projected X1 mean = {}\nprojected X2 mean = {}\npojected X1 scatter = {}\nprojected X2 scatter = {}'.format(mp1,mp2,sp1,sp2))
disc = abs(mp1 - mp2)/(sp1 + sp2)
print('discriminability: {}'.format(disc))

'''
x1_proj = np.array([v* i for i in(v.dot(x1.T))/np.linalg.norm(v)**2])
x2_proj = np.array([v* i for i in(v.dot(x2.T))/np.linalg.norm(v)**2])

plt.scatter(x1_proj[:,0],x1_proj[:,1], c = 'darkblue',label = 'X1')
plt.scatter(x2_proj[:,0],x2_proj[:,1], c = 'darkorange',label = 'X2')

plt.legend()
plt.show()
'''
