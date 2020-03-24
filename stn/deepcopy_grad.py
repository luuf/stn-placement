import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

l1 = nn.Linear(1,1)
l2 = nn.Linear(1,2)

s = nn.Sequential(l1,nn.LeakyReLU(1/3),l2)

c = copy.deepcopy(s)

params = [{'params':s.parameters()},{'params':c.parameters(), 'lr':10}]

o = optim.SGD(params=params, lr = 1)

print('s parameters')
print([p for p in s.parameters()])
print('c parameters')
print([p for p in c.parameters()])

inp = torch.normal(torch.zeros(10,1), torch.ones(10,1))
print(inp)
out = c(inp)
print(out)

ans = torch.zeros(10)
loss = F.cross_entropy(out, ans.long())
loss.backward()

print('s grad')
print([p.grad for p in s.parameters()])
print('c grad')
print([p.grad for p in c.parameters()])

o.step()

print('s parameters')
print([p for p in s.parameters()])
print('c parameters')
print([p for p in c.parameters()])
