import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from numpy.linalg import eig
from numpy.linalg import norm
from numpy.linalg import inv

train_data = np.load('C:/Users/sherw/OneDrive/Desktop/train-ubyte.npz')
test_data  = np.load('C:/Users/sherw/OneDrive/Desktop/test-ubyte.npz')

x_train, y_train = train_data['a'], train_data['b']
x_test,  y_test  = test_data['a'],  test_data['b']

x_train = np.array([x_train[i] for i in range(len(x_train)) \
                    if y_train[i] == 2 or y_train[i] ==6 ])
y_train = np.array([y_train[i] for i in range(len(y_train)) \
                    if y_train[i] == 2 or y_train[i] == 6])

x_test = np.array([x_test[i] for i in range(len(x_test)) \
                    if y_test[i] == 2 or y_test[i] ==6 ])
y_test = np.array([y_test[i] for i in range(len(y_test)) \
                    if y_test[i] == 2 or y_test[i] == 6])

for i in y_train:
    if i == 2:
        i = 0
    elif i == 6:
        i = 1
for i in y_test:
    if i == 2:
        i = 0
    elif i == 6:
        i = 1

tr_data = []
for i in range(2):
    tr_data.append([])
    
for i in range(len(x_train)):
    if y_train[i] == 2:
        c = 0
    elif y_train[i] == 6:
        c = 1
    tr_data[c].append(np.concatenate(x_train[i]))

for i in range(2):
    tr_data[i] = np.array(tr_data[i])

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

new_classes = [0,1]
new_labels = ['pants', 'eyeglasses']
num = 0
tr_proj = []
for i in range(2):
    tr_proj.append([])
for cl in new_classes:
    x_proj = np.array([[v.dot(point) for v in eig_vec] for point in tr_data[cl]])
    tr_proj[num] = x_proj
    plt.scatter(x_proj[:,0],x_proj[:,1] , label = new_labels[num],s = 0.1)
    num += 1
plt.xlabel('projected by 1st eigen-vec')
plt.ylabel('projected by 2nd eigen-vec')
plt.title('Separator and LDA on 2 classes(pants and eyeglasses)')
#plt.legend()
#plt.axis('equal')



miu2 = np.mean(tr_proj[0] , axis = 0 )
miu5 = np.mean(tr_proj[1],axis = 0)
cov2 = np.cov(tr_proj[0].T)
cov5 = np.cov(tr_proj[1].T)

sw = cov2+cov5

v = inv(sw).dot(miu2-miu5)
xaxis = np.linspace(-800,2000,3)
yaxis = xaxis*v[1]/v[0]
plt.plot(xaxis, yaxis, label = 'LDA' , c='red' , linewidth=1)
plt.axis('equal')
#plt.show()

xaxis2 = np.linspace(100,2500,3)
miu_mean = (miu2 + miu5)/2
yaxis2 = (xaxis2 - miu_mean[0])*(-v[0]/v[1]) + miu_mean[1]
plt.plot(xaxis2, yaxis2, label = 'Separator' , c='green' , linewidth=1)
plt.legend()
#plt.show()


######### Confusion Matrix
con_mat_train = np.zeros((2,2))
classes = [0,1]
for cls in classes:
    for datapoint in tr_proj[cls]:
        sign = datapoint[1]-((datapoint[0] - miu_mean[0])*(-v[0]/v[1]) + miu_mean[1])
        if sign <= 0:
            column = 0
        else:
            column = 1
        con_mat_train[cls][column] += 1
accuracy = 100*(sum([con_mat_train[i,i] for i in range(len(con_mat_train))]) / con_mat_train.sum())
print('confusion matrix for training set would be:\n',con_mat_train)
print('accuracy for training set would be:\n','{:.2f}'.format(accuracy) )
    




