import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

x = np.arange(0,1,0.001)

data = np.concatenate(np.array([i.strip().split() for i in open\
                 ("C:/Users/sherw/OneDrive/Desktop/SPR_HW3/inputs/P6/dense_data.dat").readlines()] , dtype = 'float'))

sig_val = [0.03 , 0.06]
y_total = []
sta_dev = data.std()
h = 1.06 * sta_dev * (len(data))**(-0.2)
for s in sig_val:
    y = []
    for j in x:
        pdfs = 0
        for i in data:
            pdfs += ((1/(h * len(data))) * 1/(s*sqrt(2*3.14)) * \
                         (2.718 ** -(((j-i)/h)**2/(2*(h**2)))))
        y.append(pdfs)
    y_total.append(y)
labels = ['sigma = 0.03','sigma = 0.06']
co = 0
for m in y_total:
    plt.plot(x , m , label = '{}'.format(labels[co]))
    co += 1
#plt.xlim(-2,4)
plt.xlabel('X')
plt.ylabel('PDF')
plt.title('Gaussian Kernel Density Estimation')
plt.legend()
plt.show()


'''
x2 = np.linspace(0 , 6 , 2000)
y2 = x2 * (2.718)**(-((x2)**2)/2)
#plt.plot(x2 , y2 , label = 'P2(x)')

x1 = np.linspace(0 , 6 , 2000)
y1 = []
for i in x2:
    if i <= 0:
        y1.append(0)
    elif i > 0 and i <= 1:
        y1.append(i)
    elif i > 1 and i <= 2:
        y1.append(2-i)
    else:
        y1.append(0)
y1 = np.array(y1)

plt.plot(x1 , y1 , label ='P1(x)')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.show()


### definig CDF of each distribution:
N = [10,100,1000]
iid1 = []
iid2 = []
y1 , y2 = np.array(y1) , np.array(y2)
p1 , p2 = y1 / y1.sum() , y2 / y2.sum()
#print(p1)
#print(p1.sum())
for i in N:
    iid1.append(np.around(np.random.choice(x1,i, p = p1),3))
    iid2.append(np.around(np.random.choice(x2,i, p = p2),3))


for i in range(3):
    print('generated {} iid samples from the p1(x):\n       '.format(N[i])\
          , list(iid1[i]) , end = '\n\n')
for i in range(3):
    print('generated {} iid samples from the p2(x):\n       '.format(N[i])\


#calculate h*:
h_star1 = []
h_star2 = []
counter = 0
for n in N:
    h_star1.append(1.06 * iid1[counter].var() * n**(-1/5))
    h_star2.append(1.06 * iid2[counter].var() * n**(-1/5))
    counter += 1

h_star = []
h_star.append(h_star1)
h_star.append(h_star2)


for i in range(2):
    for j in range(3):
        print('bandwidth of P{} for N of {} is {}'.format((i+1) , N[j] , h_star[i][j]))


k = [1/3 , 1 , 3]
new_h = []

for i in h_star:
    for j in i:
        new_h.append(j * np.array(k))

x = np.linspace(-10 ,10, 4000)

N_val = 0
for p in range(3):
    plt.subplot2grid((1,3),(p//3 , p%3))
    plt.plot(x2 , y2 , label = 'P2(x)')
    y_total = []
    for h in new_h[p+3]:
        y = []
        for j in x:
            pdfs = 0
            for i in iid2[p]:
                pdfs += ((1/(h * len(iid2[p]))) * 1/(sqrt(2*3.14)) * \
                         (2.718 ** -(((j-i)/h)**2/2)))
            y.append(pdfs)
        y_total.append(y)
    labels = ['h* % 3','h*','h* x 3']
    co = 0
    for m in y_total:
        plt.plot(x , m , label = '{}'.format(labels[co]))
        co += 1
    plt.xlim(-2,4)
    plt.xlabel('X')
    plt.ylabel('PDF')
    plt.title('N = {} and P2(x)'.format(N[N_val]))
    plt.legend()
    N_val += 1
plt.show()
'''

