import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import statistics as st
#np.set_printoptions(threshold=sys.maxsize)

number_arrays_tr = []
number_arrays_ts = []
number_prototype = []
for i in range(0,2):
    number_arrays_tr.append([])
    number_prototype.append([])
    number_arrays_ts.append([])
png_add_tr = "C:/Users/Sherw/Desktop/SPR_2/inputs/P4/Dataset/a/Train/%i_%.2i.png"
png_add_ts = "C:/Users/Sherw/Desktop/SPR_2/inputs/P4/Dataset/a/Test/%i_%.2i.png"

for number in range (0,2):
    for photo in range(0,20):
        photo_add_tr = png_add_tr %(number,photo+1)
        number_arrays_tr[number].append(np.array(Image.open(photo_add_tr)))
        
for number in range (0,2):
    for photo in range(0,10):
        photo_add_ts = png_add_tr %(number,photo+1)
        number_arrays_ts[number].append(np.array(Image.open(photo_add_ts)))

for i in range(0,2):
    number_prototype[i] = np.mean(number_arrays_tr[i],axis =0)
fig = plt.figure(1)
for i in range(0,2):
    plt.subplot2grid((1,2),(i//2,i%2))
    plt.imshow(number_prototype[i],cmap='gray')
    plt.xlabel('number %i' %i)

fig.suptitle("Prtotypes of 0 and 1")
plt.show()
