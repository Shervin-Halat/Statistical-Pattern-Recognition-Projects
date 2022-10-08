import numpy as np
import math
import matplotlib.pyplot as plt
a2 = 0
b2 = 6
n2 = 2000 
x2 = np.linspace(a2 , b2 , n2)
y2 = x2 * (2.718)**(-((x2)**2)/2)
plt.plot(x2 , y2 , label = 'P2(x)')

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
print(p1)
print(p1.sum())
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
for i in range(2):
    for j in range(3):
        print('bandwidth of P{} for N of {} is {}'.format((i+1) , N[j] , h_star[i][j]))

    


