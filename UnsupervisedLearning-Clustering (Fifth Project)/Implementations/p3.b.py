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


loc = 'C:/Users/sherw/OneDrive/Desktop/SPR_HW5/inputs/P3/tiny_trump_2.jpg'

im = np.array(Image.open(loc))
data = np.concatenate(im)
cl_3 = KMeans(n_clusters = 3 , random_state=0).fit(data)
cl_5 = KMeans(n_clusters = 5 , random_state=0).fit(data)
cl_7 = KMeans(n_clusters = 7 , random_state=0).fit(data)
cl_9 = KMeans(n_clusters = 9 , random_state=0).fit(data)
cl_15 = KMeans(n_clusters = 15 , random_state=0).fit(data)

label3 = cl_3.labels_
label5 = cl_5.labels_
label7 = cl_7.labels_
label9 = cl_9.labels_
label15 = cl_15.labels_

new = []
for i in label3:
    new.append(cl_3.cluster_centers_[i])
new = np.array(new,dtype = int).reshape(563,860,3)

plt.imshow(new)
plt.title('compressed image with k = 3')
plt.show()

new = []
for i in label5:
    new.append(cl_5.cluster_centers_[i])
new = np.array(new,dtype = int).reshape(563,860,3)

plt.imshow(new)
plt.title('compressed image with k = 5')
plt.show()

new = []
for i in label7:
    new.append(cl_7.cluster_centers_[i])
new = np.array(new,dtype = int).reshape(563,860,3)

plt.imshow(new)
plt.title('compressed image with k = 7')
plt.show()

new = []
for i in label9:
    new.append(cl_9.cluster_centers_[i])
new = np.array(new,dtype = int).reshape(563,860,3)

plt.imshow(new)
plt.title('compressed image with k = 9')
plt.show()

new = []
for i in label15:
    new.append(cl_15.cluster_centers_[i])
new = np.array(new,dtype = int).reshape(563,860,3)

plt.imshow(new)
plt.title('compressed image with k = 15')
plt.show()

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
