import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

x = np.arange(0,1,0.001)

data = open("C:/Users/sherw/OneDrive/Desktop/SPR_HW3/inputs/P7/animals_weight.txt").readlines()
raw_data = [x.strip().split() for x in data]
data = np.array(raw_data[1:] , dtype = 'float')

k_val = [1 , 3 , 5]

def body_w(brain_w , k):
    diff_data = np.array(data)
    diff_data[:,0] = abs(diff_data[:,0] - brain_w)
    diff_data = np.array(sorted(diff_data, key=lambda x: x[0], reverse=False))
    mean = diff_data[:,1][0:k].mean()
    return mean

test = [53.298 , 1247.122 , 0.583 , 4.859 , 0.041]


for k in k_val:
    counter = 1
    for t in test:
        z = body_w(t , k)
        print('for test sample #{} and k value of {} the predicted body weight is: {:.3f}'\
              .format(counter , k , z))
        counter += 1
