import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

file_address = 'C:/Users/Shervin/Desktop/iris.csv'

x=[]

with open(file_address) as f:
    reader = csv.reader(f)
    for i in reader:
        if i[4] == 'Iris-setosa':
            i[4] = 'red'
        elif i[4] == 'Iris-versicolor':
            i[4] = 'green'
        elif i[4] == 'Iris-virginica':
            i[4] = 'black'
        x.append(i)

x = np.array(x)
Class = x[:,4]
x = np.array(x[:,:4] ,dtype =float)


for i in range(0,3):
    x = x.astype(np.float)
    print(x[:,i] , type(x[:,i]))

plt.subplot(1,3,1)
plt.scatter(x[:,0] , x[:,3] , c= Class )
plt.xlabel("Sepal Lenght")
plt.ylabel("Sepal Width")
plt.title("Distribution of Feature 1 and 4")

plt.subplot(1,3,2)
plt.scatter(x[:,1] , x[:,2], c= Class )
plt.xlabel("Sepal Lenght")
plt.ylabel("Petal Length")
plt.title("Distribution of Feature 2 and 3")

plt.subplot(1,3,3)
plt.scatter(x[:,2] , x[:,3] , c= Class )
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.title("Distribution of Feature 3 and 4")

plt.show()

'''
T = np.array([[[1.2 , -0.3]], [[-1.8 , 0.6]], [[1.4,0.5]] , [[-0.5,-1]]])

x_new = np.array([ [x[:,0],x[:,1]] , [x[:,0],x[:,2]] , [x[:,1],x[:,3] ] ] , dtype = float)

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
        plt.xlabel("T%i" %counter)
        plt.title("Transformation of T%i" %counter )
        
plt.show()    
'''
