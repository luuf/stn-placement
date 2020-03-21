import models
import torch

loc = models.localization_dict['CNNmp']

l = []
count = 0
for i in range(1):
    m = models.model_dict['CNN']([],torch.Size((1,112,112)), loc, [], [3], False, 'scale', True, True)
    print(m.localization[0][0])
    for p in m.localization[0][0].parameters():
        if all(torch.flatten(p) == 0):
            count += 1
        else:
            l = l + list(torch.flatten(p[:]))
    print('0 count', count)
    mean = sum(l)/len(l)
    print('mean', mean)
    print('len',len(l))
    var = sum([(e-mean)**2 for e in l])/len(l)
    print('var', var)
