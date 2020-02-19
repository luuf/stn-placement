from data import get_precomputed
import numpy as np
import torch

train,test = get_precomputed('../data/svhn/extra_', normalize=True)

n = len(train.dataset)

tempn = 0
m = 0
for x,y in train:
    tempn += len(y)
    m += torch.sum(x)
    print(tempn, 'm', m / (tempn * 64**2))

m /= n * 64**2
#m = 0.2100
print('mean', m)

tempn = 0
var = 0
for x,y in train:
    tempn += len(y)
    var += torch.sum((x - m)**2)
    print(tempn, 'var', var / (tempn * 64**2))

#var = 34650238.2796
s = np.sqrt(var / (n * 64**2))
print('std', s)
