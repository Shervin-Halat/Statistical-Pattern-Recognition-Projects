import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from numpy.linalg import eig
from PIL import Image

address = 'C:/Users/sherw/OneDrive/Desktop/HW4_pattern/inputs/P6/sad_days_gray.jpg'

################################################
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
print('patch extraced image matrix: \n',im_paex)
print('patch extraced image matrix shape: \n',im_paex.shape)

for i in range (len(im_blocked)):
    im_blocked[i] = im_blocked[i].mean()
image_blocked = unblockshaped(im_blocked , 360 , 240)
plt.imshow(image_blocked , 'gray')
plt.show()
        
