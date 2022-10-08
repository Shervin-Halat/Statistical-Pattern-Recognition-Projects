import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

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

'''
plt.plot(x1 , y1 , label ='P1(x)')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.show()
'''

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

'''
for i in range(3):
    print('generated {} iid samples from the p1(x):\n       '.format(N[i])\
          , list(iid1[i]) , end = '\n\n')
for i in range(3):
    print('generated {} iid samples from the p2(x):\n       '.format(N[i])\
          , list(iid2[i]) , end = '\n\n')
'''
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

'''
for i in range(2):
    for j in range(3):
        print('bandwidth of P{} for N of {} is {}'.format((i+1) , N[j] , h_star[i][j]))
'''

k = [1/3 , 1 , 3]
new_h = []

for i in h_star:
    for j in i:
        new_h.append(j * np.array(k))
print(new_h)
#print(new_h_star)

x = np.linspace(-10 ,10, 4000)
y_total = []
for h in new_h[0]:
    print(h)
print(len(iid1[0]))
for h in new_h[5]:
    y = []
    for j in x:
        pdfs = 0
        for i in iid2[2]:
            pdfs += ((1/(h * len(iid2[2]))) * 1/(sqrt(2*3.14)) * \
                     (2.718 ** -(((j-i)/h)**2/2)))
        y.append(pdfs)
    y_total.append(y)

labels = ['h* % 3','h*','h* x 3']
co = 0
for i in y_total:
    plt.plot(x , i , label = '{}'.format(labels[co]))
    co += 1
plt.xlim(-2,4)
plt.xlabel('X')
plt.ylabel('PDF')
plt.title('gaussian kernel PDF estimation of P2(x)')
plt.legend()
plt.show()
'''

for k in range(3):
    plt.subplot2grid((1,3),(k//3 , k%3))
    for h in new_h[5]:
    y = []
    for j in x:
        pdfs = 0
        for i in iid2[2]:
            pdfs += ((1/(h * len(iid2[2]))) * 1/(sqrt(2*3.14)) * \
                     (2.718 ** -(((j-i)/h)**2/2)))
        y.append(pdfs)
    y_total.append(y)
    labels = ['h* % 3','h*','h* x 3']
    co = 0
    for i in y_total:
        plt.plot(x , i , label = '{}'.format(labels[co]))
        co += 1
    plt.xlim(-2,4)
    plt.xlabel('X')
    plt.ylabel('PDF')
    plt.title('gaussian kernel PDF estimation of P2(x)')
    plt.legend()



