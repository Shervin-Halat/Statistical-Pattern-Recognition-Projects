import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import csv

x = np.arange(0,1,0.001)

f = open("C:/Users/sherw/OneDrive/Desktop/SPR_HW3/inputs/P7/wine_quality.csv")
reader = csv.reader(f)
data = np.array(list(reader))[1:]
data = np.array([x[0].split(';') for x in data] , dtype = 'float')

def sort_dis(vec):
    dis = []
    res = np.zeros((len(data[:,11]),2))
    res[:,1] = np.array(data[:,11])
    for row in data:
        count = 0
        for i in range(len(vec)):
            count += (vec[i] - row[i]) ** 2
        dis.append(sqrt(count))
    res[:,0] = np.array(dis)
    sorted_distance = np.array(sorted(res, key=lambda x: x[0], reverse=False))
    return sorted_distance

k_val = [1 , 3 , 5]

def quality(feature , k):
    qual = sort_dis(feature)[:,1][0:k].mean()
    return qual

test  = [[7.5,0.9,0.26,2.3,0.054,19,13,0.99708,3.78,0.55,9.7],\
         [5.4,0.78,0.17,3.2,0.084,11,58,0.9987,2.94,0.83,11.8],\
         [8.2,0.56,0.46,1.7,0.069,25,15,0.997,3.39,0.65,12.5],\
         [6.0,0.7,0.01,4.6,0.093,6,104,0.99746,3.12,0.52,10.5],\
         [10.8,0.43,0.31,2.5,0.105,35,31,1.0001,3.22,0.48,11.1]]
         
for k in k_val:
    counter = 1
    for t in test:
        z = quality(t , k)
        print('for test sample #{} and k value of {} the predicted quality is: {:.2f}'\
              .format(counter , k , z))
        counter += 1

