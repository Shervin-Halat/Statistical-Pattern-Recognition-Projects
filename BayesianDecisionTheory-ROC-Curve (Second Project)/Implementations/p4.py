import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import statistics as st
#np.set_printoptions(threshold=sys.maxsize)

number_arrays_tr = []
number_arrays_ts = []
number_prototype = []
for i in range(0,10):
    number_arrays_tr.append([])
    number_prototype.append([])
    number_arrays_ts.append([])
png_add_tr = "C:/Users/Shervin/Desktop/SPR_2/inputs/P4/Dataset/c/Train/%i_%.2i.png"
png_add_ts = "C:/Users/Shervin/Desktop/SPR_2/inputs/P4/Dataset/c/Test/%i_%.2i.png"

for number in range (0,10):
    for photo in range(0,20):
        photo_add_tr = png_add_tr %(number,photo+1)
        number_arrays_tr[number].append(np.array(Image.open(photo_add_tr)))
        
for number in range (0,10):
    for photo in range(0,10):
        photo_add_ts = png_add_tr %(number,photo+1)
        number_arrays_ts[number].append(np.array(Image.open(photo_add_ts)))

for i in range(0,10):
    number_prototype[i] = np.mean(number_arrays_tr[i],axis =0)
fig = plt.figure(1)
for i in range(0,10):
    plt.subplot2grid((3,4),(i//4,i%4))
    plt.imshow(number_prototype[i],cmap='gray')

fig.suptitle("Prtotypes")
plt.show()

