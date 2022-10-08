import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import sys
import glob
from numpy.linalg import eigh
from heapq import nlargest
from numpy.linalg import inv as inv
from random import randint


###################################### Training images pre_processing ####################
def get_all_images(folder, ext):

    all_files = []
    #Iterate through all files in folder
    for file in os.listdir(folder):
        #Get the file extension
        _,  file_ext = os.path.splitext(file)

        #If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    #Get list of all files
    return all_files

loc_bush = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Bush'
loc_bill = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Bill'
loc_collin = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Collin'
loc_putin = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Putin'
loc_jean = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Jean'
loc_laura = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Laura'
loc_luiz = 'C:/Users/sherw/OneDrive/Desktop/Dataset/Luiz'


loc = 'C:/Users/sherw/OneDrive/Desktop/Bush/George_W_Bush_0007.jpg'

bush = get_all_images(loc_bush , 'jpg')
collin = get_all_images(loc_collin , 'jpg')
bill = get_all_images(loc_bill , 'jpg')
putin = get_all_images(loc_putin , 'jpg')
jean = get_all_images(loc_jean , 'jpg')
laura = get_all_images(loc_laura , 'jpg')
luiz = get_all_images(loc_luiz , 'jpg')


### takes address of image, returns face in 100x100:
def image_faces(image_path):
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    x,y,w,h = faces[0]
    im = Image.open(image_path).convert('L')
    face_image = im.crop((x, y, x + w, y + h)).resize((60,60))
    return face_image



# implement image_faces() on list of addresses of images, returns list of normalized images(Image)
def images_normal(imagelist_path):
    normalized_image = []
    for path in imagelist_path:
        img = image_faces(path)
        normalized_image.append(img)
    return normalized_image


#x = images_normal(bush)
'''
for i in x:
    plt.imshow(i,'gray')
    plt.show()
'''

##### gets image location, returns lbp of each image:
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    #print(val)
    return val    

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    #for i in range(output_list_len):
    for i in range(1):
        i+=1
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    
def LBP(image_file):
    #img_bgr = cv2.imread(image_file)
    img_bgr = np.array(image_file)
    #height, width, channel = img_bgr.shape
    height, width = img_bgr.shape
    #img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = img_bgr
    
    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    output_list = []
    output_list.append({
        "img": img_gray,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "Gray Image",
        "type": "gray"        
    })
    output_list.append({
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })    
    output_list.append({
        "img": hist_lbp,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
    })

    #show_output(output_list)
                             
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print("LBP Program is finished")
    return img_lbp
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
'''
def PCA(im, m, n, d):
    im_blocked = blockshaped(im, m ,n)
    im_paex = (np.array([np.concatenate(i) for i in im_blocked])).T
    mean = im_paex.mean(axis=1)
    main_mean = mean.reshape(m,n)
    mean_image = im_blocked - main_mean
    im_st = (np.array([np.concatenate(i) for i in mean_image])).T
    im_cov = np.cov(im_st)
    val, vec = eig(im_cov)
    val_sort = np.sort(val)[::-1]
'''
###################################################
bush_im = images_normal(bush)
bill_im = images_normal(bill)
collin_im = images_normal(collin)
putin_im = images_normal(putin)
jean_im = images_normal(jean)
laura_im = images_normal(laura)
luiz_im = images_normal(luiz)

images = [bush_im ,bill_im,collin_im,putin_im,jean_im,laura_im,luiz_im]

bush_im = np.array([np.concatenate(LBP(i)) for i in bush_im]).T
collin_im = np.array([np.concatenate(LBP(i)) for i in collin_im]).T
bill_im = np.array([np.concatenate(LBP(i)) for i in bill_im]).T
putin_im = np.array([np.concatenate(LBP(i)) for i in putin_im]).T
jean_im = np.array([np.concatenate(LBP(i)) for i in jean_im]).T
laura_im = np.array([np.concatenate(LBP(i)) for i in laura_im]).T
luiz_im = np.array([np.concatenate(LBP(i)) for i in luiz_im]).T

############### Apply PCA to datasets above: (first 100 eigenvectors considered)
mean , cen , cov ,val , vec , last_dataset = [],[],[],[],[] , []
dataset = [bush_im,bill_im,collin_im,putin_im,jean_im,laura_im,luiz_im]

for i in range(len(dataset)):
    last_dataset.append([])
    mean = dataset[i].mean(axis = 1)
    cen = (dataset[i].T - mean).T
    cov = np.cov(cen)
    val, vec = eigh(cov)
    val = np.argsort(val)[-600:][::-1]
    vec = np.array([vec[:,i] for i in val])
    last_dataset[i] = vec.dot(cen)

###################################### END of Training images pre_processing ###########

#################Training phase:

m = []
n = len(last_dataset)
for i in range(n):
    m.append(len(last_dataset[i].T))
k = sum(m)

#initialization:
se = (1/(m[0]*n)) * sum([np.cov(i) for i in last_dataset])
smiu = (1/n) * sum([(i.mean(axis =1).reshape(600,1)).dot(i.mean(axis = 1).reshape(1,600)) for i in last_dataset])


#Iteration:
# E-step:
for i in range(8):
    emiu = []
    ee = []
    f = inv(se)
    g = -inv(m[0]*smiu + se).dot(smiu.dot(inv(se)))
    for j in range(n):
        #ee.append([])
        #emiu.append([])
        emiu.append((smiu.dot(f + m[j]*g)).dot(sum([t for t in last_dataset[j].T])))
        for l in range(len(last_dataset[j].T)):
            ee.append(se.dot(f).dot(last_dataset[j][:,l]) + se.dot(g).dot(sum([t for t in last_dataset[j].T])))
        #emiut.append([])
            
    #M-step:        
    smiu = 1/n * (sum([m.reshape(600,1).dot(m.reshape(1,600)) for m in emiu]))
    se = 1/k * (sum([n.reshape(600,1).dot(n.reshape(1,600)) for n in ee]))

#################Test phase:
'''
x1 = last_dataset[0][:,5]
x2 = last_dataset[0][:,15]
x3 = last_dataset[2][:,3]
x4 = last_dataset[3][:,8]
'''
def test(t1,t2):
    A = inv(smiu + se) - inv((smiu + se) - smiu.dot(inv(smiu + se)).dot(smiu))
    G = -inv(2*smiu + se).dot(smiu).dot(inv(se))
    #similarity:
    r = t1.T.dot(A).dot(t1) + t2.T.dot(A).dot(t2) - 2 * t1.T.dot(G).dot(t2)
    return(r)

#sim = test(x1,x2)
'''
if sim > 0:
    print('equal identity')
elif sim < 0:
    print('non-equal identity')
'''
counter = 0
for i in range(50):
    a = randint(0,6)
    b = randint(0,6)
    c = randint(0,17)
    d = randint(0,17)
    x1 = last_dataset[a][:,c]
    x2 = last_dataset[b][:,d]
    s = test(x1,x2)
    if s >= 0:
        if a == b:
            counter+=1
        plt.subplot2grid((1,2),(0,0))
        plt.imshow(images[a][c],'gray')
        plt.subplot2grid((1,2),(0,1))
        plt.imshow(images[b][d],'gray')
        plt.suptitle('equal')
    else:
        plt.subplot2grid((1,2),(0,0))
        plt.imshow(images[a][c],'gray')
        plt.subplot2grid((1,2),(0,1))
        plt.imshow(images[b][d],'gray')
        plt.suptitle('not equal')
    #plt.show()
print(counter)
