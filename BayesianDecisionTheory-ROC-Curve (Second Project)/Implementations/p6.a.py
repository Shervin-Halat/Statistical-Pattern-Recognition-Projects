import csv
import numpy as np
from statistics import mean
import math
import matplotlib.pyplot as plt

datasets = []
data = [[],[],[],[]]
sigma =[[],[]]
miu=[[],[]]
for i in range(0,4):
    file_address = "C:/Users/Shervin/Desktop/SPR_2/inputs/P6/Datasets/dataset_%i.txt" %(i+1)
    datasets.append(open(file_address))
    for line in datasets[i]:
        data[i].append(line.split())
    data[i] = np.array(data[i], dtype = float)
for i in range(0,4):
    for j in range(0,2):
        mean = data[i][:,j].mean()
        print('mean dataset %i feature %i = %f' %(i+1 , j+1 , mean))
        standard_deviation = math.sqrt(((data[i][:,j] - data[i][:,j].mean())**2).mean())
        print('standard deviation dataset %i feature %i = %f' %(i+1, j+1, standard_deviation))
        sigma[j].append(standard_deviation)
        miu[j].append(mean)

for i in range(0,4):
        print('Discriminability of data set %i =' %(i+1), '%.3f' %(abs(miu[0][i]-miu[1][i])/math.sqrt(sigma[0][i]**2+ sigma[1][i]**2)))

MIN , MAX = [] , []
