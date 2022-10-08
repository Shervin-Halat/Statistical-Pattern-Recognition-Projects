import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from numpy.linalg import eig
from numpy.linalg import norm

train_data = np.load('C:/Users/sherw/OneDrive/Desktop/train-ubyte.npz')
test_data  = np.load('C:/Users/sherw/OneDrive/Desktop/test-ubyte.npz')

x_train, y_train = train_data['a'], train_data['b']
x_test,  y_test  = test_data['a'],  test_data['b']

tr_data = []
for i in range(10):
    tr_data.append([])
    
for i in range(len(x_train)):
    tr_data[y_train[i]].append(np.concatenate(x_train[i]))

x_train_col = np.array([np.concatenate(i) for i in x_train])
#print(x_train_col)
x_mean = np.mean(x_train_col,axis =0)
x_norm = x_train_col - x_mean
x_norm = np.array(x_norm)
x_cov = np.cov(x_norm.T)
e_val, e_vec = eig(x_cov)
#print(e_val.shape)
#print(e_vec.shape)

e_val_sort = np.sort(e_val)[::-1]
eig_vec = []
for i in range(2):
    eig_vec.append([])

for j in range(2):
    eig_vec[j] = e_vec[:,np.where(np.isclose(e_val,e_val_sort[j]))[0][0]]

labels = ['cloud','sun','pants','umbrella','table','ladder'\
          ,'eyeglasses','clock','scissors','Cup']

for cl in range(10):
    x_proj = np.array([[v.dot(point) for v in eig_vec] for point in tr_data[cl]])
    plt.scatter(x_proj[:,0],x_proj[:,1] , label = labels[cl],s = 0.1)
plt.xlabel('projected by 1st eigen-vec')
plt.ylabel('projected by 2nd eigen-vec')
plt.title('PCA on data')
plt.legend()
plt.axis('equal')
plt.show()
