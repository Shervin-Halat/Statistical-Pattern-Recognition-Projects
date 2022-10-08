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
    file_address = "C:/Users/Sherw/Desktop/SPR_2/inputs/P6/Datasets/dataset_%i.txt" %(i+1)
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
print(miu,sigma)

for i in range(0,4):
        print(abs(miu[0][i]+miu[1][i])/math.sqrt(sigma[0][i]**2+ sigma[1][i]**2))

MIN , MAX = [] , []
for i in range(0,4):
    MIN.append([])
    MAX.append([])
    
def MAX(dataset):
    print(max(max(dataset[:,0]),max(dataset[:,1])))
    return max(max(dataset[:,0]),max(dataset[:,1]))
def MIN(dataset):
    print(min(min(dataset[:,0]),min(dataset[:,1])))
    return min(min(dataset[:,0]),min(dataset[:,1]))

def ROC(dataset, n):
    TPR , FPR = [] , []
    x = np.linspace(MIN(dataset) , MAX(dataset) , 100)
    for xstar in x:
        counter1 = 0
        counter2 = 0
        for data in dataset[:,1]:
            if data > xstar:
                counter1 += 1
        for data in dataset[:,0]:
            if data > xstar:
                counter2 += 1     
        TPR.append(counter1/1000)
        FPR.append(counter2/1000)
    plt.plot(FPR , TPR, label='dataset%i'%n)

for i in range(0,4):
    ROC(data[i], i+1)
plt.plot([0,1],[0,1], label = 'diagonal')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()
