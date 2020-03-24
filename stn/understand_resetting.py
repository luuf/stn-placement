import models
import torch
import numpy as np
import matplotlib.pyplot as plt

loc = models.localization_dict['CNNmp']

l = []
count = 0
for i in range(10):
    m = models.model_dict['CNN']([],torch.Size((1,112,112)), loc, [], [3], False, 'scale', True, True)
    print(m.localization[0][0])
    for p in m.localization[0][0].parameters():
        if all(torch.flatten(p) == 0):
            count += 1
        else:
            l = l + list(torch.flatten(p[:]).detach())

b = 12
hist,edges = np.histogram(l, bins=b)
print(hist)
print(edges)
plt.plot([(edges[i]+edges[i+1])/2 for i in range(b)], hist)
plt.ylim(bottom=0)
plt.show()

    #print('0 count', count)
    #mean = sum(l)/len(l)
    #print('mean', mean)
    #print('len',len(l))
    #var = sum([(e-mean)**2 for e in l])/len(l)
    #print('var', var)
