import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import statistics as st
#np.set_printoptions(threshold=sys.maxsize)

number_arrays_tr = []
number_arrays_ts = []
number_prototype = []
number_cov = []
number_arrays_tr_concan = []
number_arrays_ts_concan = []
for i in range(0,10):
    number_arrays_tr.append([])
    number_prototype.append([])
    number_arrays_ts.append([])
    number_cov.append([])
    number_arrays_tr_concan.append([])
    number_arrays_ts_concan.append([])
png_add_tr = "C:/Users/Sherw/Desktop/SPR_2/inputs/P4/Dataset/c/Train/%i_%.2i.png"
png_add_ts = "C:/Users/Sherw/Desktop/SPR_2/inputs/P4/Dataset/c/Test/%i_%.2i.png"

for number in range (0,10):
    for photo in range(0,20):
        photo_add_tr = png_add_tr %(number,photo+1)
        number_arrays_tr[number].append(np.array(Image.open(photo_add_tr)))
        number_arrays_tr_concan[number].append(np.concatenate(np.array(Image.open(photo_add_tr))))
        
for number in range (0,10):
    for photo in range(0,10):
        photo_add_ts = png_add_ts %(number,photo+1)
        number_arrays_ts[number].append(np.array(Image.open(photo_add_ts)))
        number_arrays_ts_concan[number].append(np.concatenate(np.array(Image.open(photo_add_ts))))

for i in range(0,10):
    number_prototype[i] = np.mean(number_arrays_tr[i],axis =0)
fig = plt.figure(1)
for i in range(0,10):
    plt.subplot2grid((3,4),(i//4,i%4))
    plt.imshow(number_prototype[i],cmap='gray')
    plt.xlabel('number %i' %(i))

#fig.suptitle("Prtotypes")
#plt.show()
'''
#rowvar=False!
number_cov0 = []
for i in range(0,10):
    number_cov0 = np.cov(number_arrays_tr_concan[0], rowvar=False)


print(number_cov0)
number_cov0 = np.around(number_cov0 , decimals = 2 )

#number_cov = np.array(number_cov , dtype = 'int')

def Distance(array, number):
    return ((array-np.concatenate(number_prototype[number])).transpose()\
           .dot(np.linalg.pinv(number_cov0))\
           .dot((array-np.concatenate(number_prototype[number]))))
MDC = []
#for i in range(0,1):
print(number_cov0)
x = Distance(number_arrays_ts_concan[0][0] , 0)
print(x)

print("finish")
'''
fig.suptitle("Prtotypes")
plt.show()
