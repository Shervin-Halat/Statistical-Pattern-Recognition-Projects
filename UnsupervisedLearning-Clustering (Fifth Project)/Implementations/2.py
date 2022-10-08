import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import sample
from random import randint
from math import sqrt


a = [[1,3],[2,1],[2,5],[4,1],[4,5],[5,3]]

for i in range(len(a)):
    plt.scatter(a[i][0],a[i][1],label = '{}'.format(i+1))
plt.legend()
plt.show()

'''
data = []
r = 0
i = 0
j = 0
for i in range(30):
    r += 0.1
    x = r
    for j in range(30):
        j+=0.1
        data.append([i,j])
data = np.array(data)


def kmeans(dataset , k):

    #center = sample(list(dataset), k)
    label_data = []
    temp = []
    center = []
    
    for n in range(k):
        label_data.append([])
        center.append([])
        center[n] = data[randint(0,len(data)-1)]

    while True:
        for i in dataset:
            dist = []
            for c in center:
                dist.append(np.linalg.norm(i - c))
            label_data[dist.index(min(dist))].append(list(i))

        if temp == label_data:
            break
            
        temp = list(label_data)
        
        for i in range(k):
            center[i] = np.mean(label_data[i],axis = 0)
            label_data[i] = []
        #label_data = [[]]*k

    for i in range(len(label_data)):
        label_data[i] = np.array(label_data[i])

    return label_data
for i in range(20):
    l = kmeans(data,2)
    plt.scatter(l[0][:,0],l[0][:,1],s = 2)
    plt.scatter(l[1][:,0],l[1][:,1],s = 2)
    plt.show()
'''
