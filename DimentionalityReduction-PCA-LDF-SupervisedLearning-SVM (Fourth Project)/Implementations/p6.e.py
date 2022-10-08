import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from numpy.linalg import eig
from PIL import Image

address = 'C:/Users/sherw/OneDrive/Desktop/HW4_pattern/inputs/P6/sad_days_gray.jpg'

#################################################
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

#################################################

im = np.array(Image.open(address).convert('L'))

im_blocked = blockshaped(im, 8 ,8)
im_paex = (np.array([np.concatenate(i) for i in im_blocked])).T
#print('patch extraced image matrix: \n',im_paex)
#print('patch extraced image matrix shape: \n',im_paex.shape)

mean = im_paex.mean(axis=1)
main_mean = mean.reshape(8,8)
mean_image = im_blocked - main_mean
mean2 = unblockshaped(mean_image, 360, 240)
im_st = (np.array([np.concatenate(i) for i in mean_image])).T

plt.imshow(main_mean)

im_cov = np.cov(im_st)

val, vec = eig(im_cov)
val_sort = np.sort(val)[::-1]
'''
for i in range (20):
    print('{}st biggest eigen-value is: \n{}'.format(i+1,val_sort[i]) \
          , '\ncorresponding eigen-vector is:\n {}'.format(vec[np.where(np.isclose(val,val_sort[i]))[0][0]]))
'''

'''
for i in range (8):
    eig_im = vec[np.where(np.isclose(val,val_sort[i]))[0][0]]
    plt.subplot2grid((2,4), (i//4, i%4))
    eig_im = eig_im.reshape(8,8)
    #eig_img = Image.fromarray(eig_im , 'RGB')
    plt.imshow(eig_im,'gray')
    plt.title('eig-image of {}st biggest eig-value'.format(i+1))
plt.show()
'''
'''
k = [2,5,10,20]

i = 0
for value in k:
    plt.subplot2grid((2,2),(i//2,i%2))
    i+=1
    eig_im = vec[np.where(np.isclose(val,val_sort[value]))[0][0]]
    a = (eig_im.T).dot(im_st)
    print(a.shape)
    plt.imshow(a.reshape(45,30),'gray')
    plt.title('reconstructed subimage for k = {}'.format(value))
plt.show()
'''
####################################################

b = vec[np.where(np.isclose(val,val_sort[2]))[0][0]]
c = (b.T).dot(im_st)
d = np.array([b*i for i in c])
d = d.T
print(d.shape)
#print(d)
e = np.array([d[:,i].reshape(8,8) for i in range (d.shape[1])])
#print(e)
#print(e.shape)
z = unblockshaped(e, 360, 240)
plt.imshow(z)
plt.show()

###
k = [2,5,10,20]
k = range(64)
dd = np.zeros((1350,64))
for v in k:
    b = vec[np.where(np.isclose(val,val_sort[v]))[0][0]]
    c = (b.T).dot(im_st)
    d = np.array([b*i for i in c])
    dd += d
    
    
dd = dd.T 
e = np.array([dd[:,i].reshape(8,8) for i in range (dd.shape[1])])
z = unblockshaped(e, 360, 240)
plt.imshow(z,'gray')
plt.title('merge of reconstructed subimages for ALL k values')
plt.show() 
