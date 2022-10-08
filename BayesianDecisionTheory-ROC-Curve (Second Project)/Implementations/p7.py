from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


mask_list = []
mask_address = 'C:/Users/Shervin/Desktop/SPR_2/inputs/P7/Dataset/Train/Masks/*.png'

mask_list = list(map(np.array , list(map(Image.open , glob(mask_address)))))

all_pixels , black_pixels = 0 , 0
for i in range(len(mask_list):
               
for mask in mask_list:
    for pixel in mask:
        print(pixel)
        if pixel == 0:
            black_pixels += 1
    all_pixels += mask.size
prob_face = black_pixels / all_pixels
print(prob_face)
               
