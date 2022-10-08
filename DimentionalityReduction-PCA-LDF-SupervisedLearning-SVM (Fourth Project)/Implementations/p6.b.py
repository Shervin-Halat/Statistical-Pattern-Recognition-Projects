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
ii = np.array(im)

im_blocked = blockshaped(im, 8 ,8)
im_paex = (np.array([np.concatenate(i) for i in im_blocked])).T
#print('patch extraced image matrix: \n',im_paex)
#print('patch extraced image matrix shape: \n',im_paex.shape)

mean = im_paex.mean(axis=1)
main_mean = mean.reshape(8,8)
mean_image = im_blocked - main_mean
mean2 = unblockshaped(mean_image, 360, 240)
mean3 = (np.array([np.concatenate(i) for i in mean_image])).T
print(mean3.shape)

plt.imshow(main_mean)

im_cov = np.cov(mean3)
print(im_cov.shape)

val, vec = eig(im_cov)
val_sort = np.sort(val)[::-1]

for i in range (20):
    print('{}st biggest eigen-value is: \n{}'.format(i+1,val_sort[i]) \
          , '\ncorresponding eigen-vector is:\n {}'.format(vec[np.where(np.isclose(val,val_sort[i]))[0][0]]))

mean = mean.reshape(8,8)


img = Image.fromarray(mean)
plt.imshow(mean,'gray')
plt.title('mean image')
plt.show()

for i in range (8):
    eig_im = vec[np.where(np.isclose(val,val_sort[i]))[0][0]]
    plt.subplot2grid((2,4), (i//4, i%4))
    eig_im = eig_im.reshape(8,8)
    #eig_img = Image.fromarray(eig_im , 'RGB')
    plt.imshow(eig_im,'gray')
    plt.title('eig-image of {}st biggest eig-value'.format(i+1))
plt.show()

mean2 = im_blocked - mean
mean2 = np.reshape(mean2,(360,240))
plt.imshow(mean2,'gray')
plt.show()
