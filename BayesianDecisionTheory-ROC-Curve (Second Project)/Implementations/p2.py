import numpy as np

x = np.array([[[87],[62]],[[156],[36]],[[19],[3]],[[132],[3]],[[62],[49]]])
miu_silver = np.array([[101.2],[1.5]])
miu_bronze = np.array([[105.8],[41.8]])
miu_gold = np.array([[33],[51.2]])

for i in range(0,5):
    g_silver , g_bronze , g_gold , g = [] , [] , [] , []
    g_silver.append(-((x[i]-miu_silver).transpose()).dot(x[i]-miu_silver)[0][0]/2)
    g_bronze.append(-((x[i]-miu_bronze).transpose()).dot(x[i]-miu_bronze)[0][0]/2)
    g_gold.append(-((x[i]-miu_gold).transpose()).dot(x[i]-miu_gold)[0][0]/2)
    g_silver.append('silver')
    g_bronze.append('bronze')
    g_gold.append('gold')
    g.append(g_gold)
    g.append(g_silver)
    g.append(g_bronze)
    g = sorted(g, key = lambda x: x[0] , reverse=True)
    print('Pixel %i: ' %(i+1),'G(silver)=', '%.2f' %g_silver[0],\
          'G(bronze)=' , '%.2f' %g_bronze[0], 'G(gold)=' ,'%.2f' %g_gold[0])
    
    print('Hence, for pixel %i the appropriate Class is ' %(i+1), g[0][1] ,'\n')
