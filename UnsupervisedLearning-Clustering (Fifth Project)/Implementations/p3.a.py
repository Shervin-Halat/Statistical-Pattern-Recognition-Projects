import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import sample
from random import randint
from math import sqrt
from scipy import spatial
from PIL import Image
from sklearn.cluster import KMeans


loc = 'C:/Users/sherw/OneDrive/Desktop/SPR_HW5/inputs/P3/tiny_trump_1.jpg'

im = np.array(Image.open(loc))
data = np.concatenate(im)
cl_3 = KMeans(n_clusters = 3 , random_state=0).fit(data)
cl_5 = KMeans(n_clusters = 5 , random_state=0).fit(data)
cl_7 = KMeans(n_clusters = 7 , random_state=0).fit(data)
cl_9 = KMeans(n_clusters = 9 , random_state=0).fit(data)

gr = [cl_3,cl_5,cl_7,cl_9]
counter = 1
for i in gr:
    counter += 2
    print('main {} colors in RGB is:'.format(counter))
    print(i.cluster_centers_)
    
colors = []

for i in cl_3.cluster_centers_:
    for j in range(10000):
        colors.append(i)
colors = np.array(colors,dtype = int).reshape(300,100,3)
plt.imshow(colors)
plt.title('main 3 colors:')
plt.show()

for i in cl_5.cluster_centers_:
    for j in range(10000):
        colors.append(i)
colors = np.array(colors,dtype = int).reshape(500,100,3)
plt.imshow(colors)
plt.title('main 5 colors:')
plt.show()

for i in cl_7.cluster_centers_:
    for j in range(10000):
        colors.append(i)
colors = np.array(colors,dtype = int).reshape(700,100,3)
plt.imshow(colors)
plt.title('main 7 colors:')
plt.show()

for i in cl_9.cluster_centers_:
    for j in range(10000):
        colors.append(i)
colors = np.array(colors,dtype = int).reshape(900,100,3)
plt.imshow(colors)
plt.title('main 9 colors:')
plt.show()

#for i in range(
# implemented Kmeans
'''
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

clusters = kmeans(data,3)

print(clusters)
'''
