import matplotlib.pyplot as plt
import csv
import numpy as np
from statistics import mean
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

file_address = 'C:/Users/Shervin/Desktop/first_half_logs.csv'

x=[]

with open(file_address) as f:
    reader = csv.reader(f)
    for i in reader:
        x.append(i)
        
IDs=[]
for i in x:
    if i[1] in IDs:
        continue
    else:
        IDs.append(i[1])
    if len(IDs) == 11:
        break
IDs_str = IDs
IDs  = list(map(int , IDs))
IDs.sort()
mean_loc = [[]]*11
counter=0
Location = [[]]*11
#print(len(IDs) , len(mean_loc))
for i in IDs:
    xy = []
    for j in x:
        if i == int(j[1]):
            xy.append([float(j[2]),float(j[3])])
    mean_loc[counter] = [mean([k[0] for k in xy]) , mean([k[1] for k in xy])]
    Location [counter] = xy
#    print(mean_loc)
    counter += 1
#print(mean_loc)
mean_loc_np = np.array(mean_loc)
for i,txt in enumerate(IDs_str):
    plt.figure(1)
    plt.scatter(mean_loc_np[:,0][i],mean_loc_np[:,1][i])
    plt.text(mean_loc_np[:,0][i]+0.5,mean_loc_np[:,1][i]+0.5,txt)
    plt.xlim(0,105)
    plt.ylim(0,75)
#plt.show()

xy = np.array(xy)
cov = [[]]*11
norm_dis = [[]]*11
for i in range(0,11):
    cov[i] = np.cov([k[0] for k in Location[i]] , [k[1] for k in Location[i]] )
for j in range(0,11):
    norm_dis[j] = multivariate_normal(mean_loc[j],cov[j])
for i in range(0,11):
    print("mean and covariance of player ID %i is:" %IDs[i] , mean_loc[i] , cov[i])
                                                
#############################################
x = np.linspace(0,105,500)
y = np.linspace(0,75,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
for i in range(0,11):
    rv = multivariate_normal([mean_loc[i][0],mean_loc[i][1] ],cov[i] )
    fig = plt.figure(i+2)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='plasma',linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('norm_dis_pdf axis')
plt.show()

'''x = np.array(x)
Class = x[:,4]
x = np.array(x[:,:4] ,dtype =float)

for i in range(0,3):
    x = x.astype(np.float)
    print(x[:,i] , type(x[:,i]))

plt.subplot(1,3,1)
plt.scatter(x[:,0] , x[:,1] , c= Class )


plt.subplot(1,3,2)
plt.scatter(x[:,0] , x[:,2], c= Class )


plt.subplot(1,3,3)
plt.scatter(x[:,1] , x[:,3] , c= Class )


plt.show()

T = np.array([[[1 , 4]], [[-3 , 3]], [[1.4,0.5]] , [[-0.5,-1]]])

x_new = np.array([ [x[:,0],x[:,3]] , [x[:,1],x[:,2]] , [x[:,2],x[:,3] ] ] , dtype = float)

###
y=[]
for i in range(0,3):
    for j in range(0,50):
        y.append(i)
###
    
counter2=0
for i in x_new:
    counter=0
    counter2 += 1 
    for j in T:
        plt.figure(counter2)
        counter += 1
        z = j.dot(i)
        print(z)
        plt.subplot(1,4, counter)
        plt.scatter(z , y, s=10 ,facecolors='none',edgecolors=Class )
plt.show()    
'''
