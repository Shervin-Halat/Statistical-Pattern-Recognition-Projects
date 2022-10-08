import numpy as np
import math
import matplotlib.pyplot as plt

x1 = np.linspace(0 , 6 , 200)
y1 = x1 * (2.718)**(-((x1)**2)/2)
plt.plot(x1 , y1 , label = 'P2(x)')

x2 = np.linspace(0 , 6 , 200)
y2 = []
for i in x2:
    if i <= 0:
        y2.append(0)
    elif i > 0 and i <= 1:
        y2.append(i)
    elif i > 1 and i <= 2:
        y2.append(2-i)
    else:
        y2.append(0)
y2 = np.array(y2)
plt.plot(x2 , y2 , label ='P1(x)')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.show()

