import numpy as np
from math import sqrt


h = 4
x = [4,9,16]
z = [2,3,5,6,6,7,8,8,8,11,12,12,14,18,20,20]
y = []

for j in x:
    pdfs = 0
    for i in z:
        pdfs += ((1/(h * len(z))) * 1/(sqrt(2*3.14)) * \
                    (2.718 ** -(((j-i)/h)**2/2)))
    y.append(pdfs)
print(y)
