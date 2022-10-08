import matplotlib.pyplot as plt
import csv
import numpy as np
from statistics import mean

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
#print(len(IDs) , len(mean_loc))
for i in IDs:
    xy = []
    for j in x:
        if i == int(j[1]):
            xy.append([float(j[2]),float(j[3])])
    mean_loc[counter] = [mean([k[0] for k in xy]) , mean([k[1] for k in xy])]
#    print(mean_loc)
    counter += 1
print(mean_loc)
mean_loc = np.array(mean_loc)
for i,txt in enumerate(IDs_str):
    plt.scatter(mean_loc[:,0][i],mean_loc[:,1][i])
    plt.text(mean_loc[:,0][i]+0.5,mean_loc[:,1][i]+0.5,txt)
#plt.scatter(mean_loc[:,0],mean_loc[:,1], text = IDs_str)
plt.xlim(0,105)
plt.ylim(0,75)
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
